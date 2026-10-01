// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
#include <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <os/availability.h>
#pragma clang diagnostic pop

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "LayerShapeConstraints.hpp"

// Defined here so every source file sees it: the int8 MLMultiArray conversions in
// CoreMLPythonArray.mm and CoreMLPythonUtils.mm depend on it.
#ifndef BUILT_WITH_MACOS26_SDK
#if TARGET_OS_OSX && defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
#define BUILT_WITH_MACOS26_SDK 1
#else
#define BUILT_WITH_MACOS26_SDK 0
#endif
#endif

namespace py = pybind11;

namespace CoreML {
    namespace Python {
        namespace Utils {

            bool isCompiledModelPath(const std::string& path);
            NSURL * stringToNSURL(const std::string& str);
            void handleError(NSError *error);
            
            // python -> objc
            MLDictionaryFeatureProvider * dictToFeatures(const py::dict& dict, NSError **error);
            MLFeatureValue * convertValueToObjC(const py::handle& handle);
            
            // objc -> cpp
            std::vector<size_t> convertNSArrayToCpp(NSArray<NSNumber *> *array);
            NSArray<NSNumber *>* convertCppArrayToObjC(const std::vector<size_t>& array);
            
            // objc -> python
            py::dict featuresToDict(id<MLFeatureProvider> features);
            py::object convertValueToPython(MLFeatureValue *value);
            py::object convertArrayValueToPython(MLMultiArray *value);
            py::object convertDictionaryValueToPython(NSDictionary<NSObject *,NSNumber *> * value);
            py::object convertImageValueToPython(CVPixelBufferRef value);
            py::object convertSequenceValueToPython(MLSequence *seq) API_AVAILABLE(macos(10.14));
        }
    }
}
