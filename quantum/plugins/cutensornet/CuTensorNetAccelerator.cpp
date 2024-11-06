/*******************************************************************************
 * Copyright (c) 2024 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Daniel Claudino - initial API and implementation
 *******************************************************************************/
#include "CuTensorNetAccelerator.hpp"
#include "xacc_plugin.hpp"

namespace xacc {
namespace quantum {

int64_t CuTensorNetVisitor::id = 0;

void CuTensorNetAccelerator::initialize(const HeterogeneousMap &params) {

  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (!cutnHandle) {
    HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  }
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  if (params.keyExists<int>("bond-dimension")) {
    bondDimension = params.get<int>("bond-dimension");
  }

  return;
}

void CuTensorNetAccelerator::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::shared_ptr<CompositeInstruction> program) {

  auto numQubits = buffer->size();
  // Determine the MPS representation and allocate buffers for the MPS tensors
  const int64_t maxExtent =
      bondDimension; // GHZ state can be exactly represented with max bond
                     // dimension of 2
  std::vector<std::vector<int64_t>> extents;
  std::vector<int64_t *> extentsPtr(numQubits);
  std::vector<void *> d_mpsTensors(numQubits, nullptr);
  for (int32_t i = 0; i < numQubits; i++) {
    if (i == 0) { // left boundary MPS tensor
      extents.push_back({2, maxExtent});
      HANDLE_CUDA_ERROR(
          cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * 2 * fp64size));
    } else if (i == numQubits - 1) { // right boundary MPS tensor
      extents.push_back({maxExtent, 2});
      HANDLE_CUDA_ERROR(
          cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * 2 * fp64size));
    } else { // middle MPS tensors
      extents.push_back({maxExtent, 2, maxExtent});
      HANDLE_CUDA_ERROR(cudaMalloc(&d_mpsTensors[i],
                                   2 * maxExtent * maxExtent * 2 * fp64size));
    }
    extentsPtr[i] = extents[i].data();
  }

  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize =
      (freeSize - (freeSize % 4096)) /
      2; // use half of available memory with alignment
  void *d_scratch{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize
            << " bytes of scratch memory on GPU\n";

  // Create the initial quantum state
  cutensornetState_t quantumState;
  const std::vector<int64_t> qubitDims(numQubits, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
      CUDA_C_64F, &quantumState));

  CuTensorNetVisitor visitor(quantumState, cutnHandle);
  InstructionIterator it(program);
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled()) {
      if (nextInst->name() != "Measure") {
        nextInst->accept(&visitor);
      } else {
        // Just collect the indices of measured qubit
        // measureBitIdxs.emplace_back(nextInst->bits()[0]);
      }
    }
  }
  std::cout << "Applied quantum gates\n";
  // Specify the final target MPS representation (use default fortran strides)
  HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
      cutnHandle, quantumState, CUTENSORNET_BOUNDARY_CONDITION_OPEN,
      extentsPtr.data(), /*strides=*/nullptr));

  // Optional, set up the SVD method for truncation.
  cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      cutnHandle, quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO, &algo,
      sizeof(algo)));
  std::cout << "Configured the MPS computation\n";

  // Prepare the MPS computation and attach workspace
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  std::cout << "Created the workspace descriptor\n";
  // exit(0);
  HANDLE_CUTN_ERROR(cutensornetStatePrepare(cutnHandle, quantumState,
                                            scratchSize, workDesc, 0x0));
  std::cout << "Prepared the computation of the quantum circuit state\n";
  double flops{0.0};
  HANDLE_CUTN_ERROR(cutensornetStateGetInfo(cutnHandle, quantumState,
                                            CUTENSORNET_STATE_INFO_FLOPS,
                                            &flops, sizeof(flops)));
  if (flops > 0.0) {
    std::cout << "Total flop count = " << (flops / 1e9) << " GFlop\n";
  } else if (flops < 0.0) {
    std::cout << "ERROR: Negative Flop count!\n";
    std::abort();
  }

  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  std::cout << "Scratch GPU workspace size (bytes) for MPS computation = "
            << worksize << std::endl;
  if (worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  } else {
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer for MPS computation\n";

  // Execute MPS computation
  HANDLE_CUTN_ERROR(cutensornetStateCompute(
      cutnHandle, quantumState, workDesc, extentsPtr.data(),
      /*strides=*/nullptr, d_mpsTensors.data(), 0));

  // Create an empty tensor network operator
  cutensornetNetworkOperator_t hamiltonian;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(
      cutnHandle, numQubits, qubitDims.data(), CUDA_C_64F, &hamiltonian));
  // Append component (0.5 * Z1 * Z2) to the tensor network operator
  {
    const int32_t numModes[] = {1, 1}; // Z1 acts on 1 mode, Z2 acts on 1 mode
    const int32_t modesX0[] = {0};     // state modes Z1 acts on
    const int32_t modesX1[] = {1};     // state modes Z2 acts on
    const int32_t *stateModes[] = {modesX0,
                                   modesX1}; // state modes (Z1 * Z2) acts on
    const void *gateData[] = {visitor.getX(),
                              visitor.getX()}; // GPU pointers to gate data
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(
        cutnHandle, hamiltonian, cuDoubleComplex{1.0, 0.0}, 2, numModes,
        stateModes, NULL, gateData, &CuTensorNetVisitor::id));
  }
  /*
  // Append component (0.25 * Y3) to the tensor network operator
  {
    const int32_t numModes[] = {1}; // Y3 acts on 1 mode
    const int32_t modesY3[] = {3}; // state modes Y3 acts on
    const int32_t * stateModes[] = {modesY3}; // state modes (Y3) acts on
    const void * gateData[] = {visitor.getY()}; // GPU pointers to gate data
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle,
  hamiltonian, cuDoubleComplex{0.25,0.0}, 1, numModes, stateModes, NULL,
  gateData, &CuTensorNetVisitor::id));
  }
  // Append component (0.13 * Y0 X2 Z3) to the tensor network operator
  {
    const int32_t numModes[] = {1, 1, 1}; // Y0 acts on 1 mode, X2 acts on 1
  mode, Z3 acts on 1 mode const int32_t modesY0[] = {0}; // state modes Y0 acts
  on const int32_t modesX2[] = {2}; // state modes X2 acts on const int32_t
  modesZ3[] = {3}; // state modes Z3 acts on const int32_t * stateModes[] =
  {modesY0, modesX2, modesZ3}; // state modes (Y0 * X2 * Z3) acts on const void
  * gateData[] = {visitor.getY(), visitor.getX(), visitor.getZ()}; // GPU
  pointers to gate data
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle,
  hamiltonian, cuDoubleComplex{0.13,0.0}, 3, numModes, stateModes, NULL,
  gateData, &CuTensorNetVisitor::id));
  }
  */
  std::cout << "Constructed a tensor network operator: (0.5 * Z1 * Z2) + (0.25 "
               "* Y3) + (0.13 * Y0 * X2 * Z3)"
            << std::endl;

  // Specify the quantum circuit expectation value
  cutensornetStateExpectation_t expectation;
  HANDLE_CUTN_ERROR(cutensornetCreateExpectation(cutnHandle, quantumState,
                                                 hamiltonian, &expectation));
  std::cout << "Created the specified quantum circuit expectation value\n";

  // Configure the computation of the specified quantum circuit expectation
  // value
  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(
      cutnHandle, expectation, CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES,
      &numHyperSamples, sizeof(numHyperSamples)));

  // Prepare the specified quantum circuit expectation value for computation
  HANDLE_CUTN_ERROR(cutensornetExpectationPrepare(cutnHandle, expectation,
                                                  scratchSize, workDesc, 0x0));
  std::cout << "Prepared the specified quantum circuit expectation value\n";
  flops = 0.0;
  HANDLE_CUTN_ERROR(cutensornetExpectationGetInfo(
      cutnHandle, expectation, CUTENSORNET_EXPECTATION_INFO_FLOPS, &flops,
      sizeof(flops)));
  std::cout << "Total flop count = " << (flops / 1e9) << " GFlop\n";
  if (flops <= 0.0) {
    std::cout << "ERROR: Invalid Flop count!\n";
    std::abort();
  }

  // Attach the workspace buffer
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  std::cout << "Required scratch GPU workspace size (bytes) = " << worksize
            << std::endl;
  if (worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  } else {
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer\n";

  // Compute the specified quantum circuit expectation value
  std::complex<double> expectVal{0.0, 0.0}, stateNorm2{0.0, 0.0};
  HANDLE_CUTN_ERROR(cutensornetExpectationCompute(
      cutnHandle, expectation, workDesc, static_cast<void *>(&expectVal),
      static_cast<void *>(&stateNorm2), 0x0));
  std::cout << "Computed the specified quantum circuit expectation value\n";
  expectVal /= stateNorm2;
  std::cout << "Expectation value = (" << expectVal.real() << ", "
            << expectVal.imag() << ")\n";
  std::cout << "Squared 2-norm of the state = (" << stateNorm2.real() << ", "
            << stateNorm2.imag() << ")\n";
  buffer->addExtraInfo("exp-val-z", expectVal.real());

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit expectation value
  HANDLE_CUTN_ERROR(cutensornetDestroyExpectation(expectation));
  std::cout << "Destroyed the quantum circuit state expectation value\n";

  // Destroy the tensor network operator
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkOperator(hamiltonian));
  std::cout << "Destroyed the tensor network operator\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  for (int32_t i = 0; i < numQubits; i++) {
    HANDLE_CUDA_ERROR(cudaFree(d_mpsTensors[i]));
  }
  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  std::cout << "Freed memory on GPU\n";

  return;
}

void CuTensorNetAccelerator::execute(
    std::shared_ptr<AcceleratorBuffer> buffer,
    const std::vector<std::shared_ptr<CompositeInstruction>>
        compositeInstruction) {

  return;
}

} // namespace quantum
} // namespace xacc
REGISTER_ACCELERATOR(xacc::quantum::CuTensorNetAccelerator)
