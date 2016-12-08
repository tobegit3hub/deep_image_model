#!/usr/bin/env python2.7

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf
import time

# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2

import predict_pb2
import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer("benchmark_test_number", 1, "")
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    data = f.read()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))


    request_number = FLAGS.benchmark_test_number
    start_time = time.time()

    for i in range(request_number):
      result = stub.Predict(request, 10.0)  # 10 secs timeout
      # print(result)

    end_time = time.time()
    print("Average latency is: {} ms".format((end_time - start_time) * 1000 / request_number))

if __name__ == '__main__':
  tf.app.run()
