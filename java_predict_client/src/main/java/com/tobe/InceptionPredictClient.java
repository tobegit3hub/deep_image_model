package com.tobe;


import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The inception predict client for TensorFlow models.
 */
public class InceptionPredictClient {
    private static final Logger logger = Logger.getLogger(InceptionPredictClient.class.getName());
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    // Initialize gRPC client
    public InceptionPredictClient(String host, int port) {
        channel = ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true)
                .build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    public static void main(String[] args) {
        System.out.println("Start the predict client");

        String host = "127.0.0.1";
        int port = 9000;
        String modelName = "cancer";
        long modelVersion = -1;

        // Parse command-line arguments
        if (args.length == 4) {
            host = args[0];
            port = Integer.parseInt(args[1]);
            modelName = args[2];
            modelVersion = Long.parseLong(args[3]);
        }

        // Run predict client to send request
        InceptionPredictClient client = new InceptionPredictClient(host, port);

        try {
            client.do_predict(modelName, modelVersion);
        } catch (Exception e) {
            System.out.println(e);
        } finally {
            try {
                client.shutdown();
            } catch (Exception e) {
                System.out.println(e);
            }
        }

        System.out.println("End of predict client");
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void do_predict(String modelName, long modelVersion) {

        // Generate features TensorProto
        TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();

        String imagePath = "../data/inference/benchmark_image.jpg";
        logger.info("Start to convert the image: " + imagePath);

        try {
            InputStream imageStream = new FileInputStream(imagePath);
            ByteString imageData = ByteString.readFrom(imageStream);

            featuresTensorBuilder.addStringVal(imageData);

            imageStream.close();
        } catch (IOException e) {
            logger.log(Level.WARNING, e.getMessage());
            System.exit(1);
        }

        TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).build();
        featuresTensorBuilder.setDtype(DataType.DT_STRING).setTensorShape(featuresShape);
        TensorProto featuresTensorProto = featuresTensorBuilder.build();


        // Generate gRPC request
        Model.ModelSpec.Builder modelTensorBuilder = Model.ModelSpec.newBuilder().setName(modelName);
        if (modelVersion > 0) {
            com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
            modelTensorBuilder.setVersion(version);
        }
        Model.ModelSpec modelSpec = modelTensorBuilder.build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec).putInputs("images", featuresTensorProto).build();

        // Request gRPC server
        Predict.PredictResponse response;
        try {
            response = blockingStub.predict(request);
            java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputs();
            for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : outputs.entrySet()) {
                System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
            }
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }
    }

}
