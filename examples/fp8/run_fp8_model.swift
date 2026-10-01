//  Copyright (c) 2026, Apple Inc. All rights reserved.
//
//  Use of this source code is governed by a BSD-3-clause license that can be
//  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

// Runs a compiled Core ML model (.mlmodelc) with plain Core ML, no coremltools:
// prints where each op runs, then times predictions on the CPU and on the Neural Engine.
//
//     swift run_fp8_model.swift models/conv512x32_fp8_w8a8.mlmodelc

import CoreML
import Foundation

func deviceName(_ device: MLComputeDevice?) -> String {
    switch device {
    case .some(.neuralEngine): return "NeuralEngine"
    case .some(.gpu): return "GPU"
    case .some(.cpu): return "CPU"
    default: return "-"
    }
}

func randomInputs(_ model: MLModel) throws -> MLFeatureProvider {
    var features: [String: MLFeatureValue] = [:]
    for (name, description) in model.modelDescription.inputDescriptionsByName {
        guard let constraint = description.multiArrayConstraint else { continue }
        let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
        for i in 0..<array.count {
            array[i] = NSNumber(value: Float.random(in: -1...1))
        }
        features[name] = MLFeatureValue(multiArray: array)
    }
    return try MLDictionaryFeatureProvider(dictionary: features)
}

func maxAbs(_ output: MLFeatureProvider) -> Float {
    var result: Float = 0
    for name in output.featureNames {
        guard let array = output.featureValue(for: name)?.multiArrayValue else { continue }
        for i in 0..<array.count {
            result = max(result, abs(array[i].floatValue))
        }
    }
    return result
}

@main
struct RunFP8Model {
    static func main() async throws {
        guard CommandLine.arguments.count > 1 else {
            print("usage: swift run_fp8_model.swift <model.mlmodelc>")
            return
        }
        let url = URL(fileURLWithPath: CommandLine.arguments[1])

        let planConfig = MLModelConfiguration()
        planConfig.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: planConfig)
        if case let .program(program) = plan.modelStructure,
           let main = program.functions["main"] {
            var counts: [String: Int] = [:]
            for op in main.block.operations where op.operatorName != "const" {
                guard let usage = plan.deviceUsage(for: op) else { continue }
                counts["\(op.operatorName) -> \(deviceName(usage.preferred))", default: 0] += 1
            }
            for (key, count) in counts.sorted(by: { $0.key < $1.key }) {
                print("  \(key) x\(count)")
            }
        }

        for (label, units) in [("CPU", MLComputeUnits.cpuOnly), ("CPU+NE", .cpuAndNeuralEngine)] {
            let config = MLModelConfiguration()
            config.computeUnits = units
            let model = try MLModel(contentsOf: url, configuration: config)
            let input = try randomInputs(model)
            let output = try await model.prediction(from: input)
            for _ in 0..<5 { _ = try await model.prediction(from: input) }
            var times: [Double] = []
            for _ in 0..<50 {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try await model.prediction(from: input)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            times.sort()
            print(String(format: "  %-7@ median %.3f ms, max|y| %.3f", label, times[times.count / 2], maxAbs(output)))
        }
    }
}
