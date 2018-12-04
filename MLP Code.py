import input_data
import matplotlib.pyplot as pp
import tensorflow as tf

numberPixelDatasets = input_data.read_data_sets("MNIST_data/", one_hot = True)

learningRate       = 0.001
trainingEpochCount = 20
batchSize          = 100
displayStep        = 1

inputNodeCount   = 784 # 입력 노드 카운트 ; 28×28 픽셀 이미지
hiddenNodeCount1 = 256 # 은닉 노드 카운트 1
hiddenNodeCount2 = 256 # 은닉 노드 카운트 2
outputNodeCount  = 10  # 출력 노드 카운트 ; 슷자 0-9

inputValueTensor  = tf.placeholder("float", [None, inputNodeCount])
outputValueTensor = tf.placeholder("float", [None, outputNodeCount])

hiddenLayerWeightVariable1 = tf.Variable(tf.random_normal([inputNodeCount, hiddenNodeCount1]))
hiddenLayerBiasVariable1   = tf.Variable(tf.random_normal([hiddenNodeCount1]))
hiddenLayerTensor1         = tf.nn.sigmoid(tf.add(tf.matmul(inputValueTensor, hiddenLayerWeightVariable1), hiddenLayerBiasVariable1))

hiddenLayerWeightVariable2 = tf.Variable(tf.random_normal([hiddenNodeCount1, hiddenNodeCount2]))
hiddenLayerBiasVariable2   = tf.Variable(tf.random_normal([hiddenNodeCount2]))
hiddenLayerTensor2         = tf.nn.sigmoid(tf.add(tf.matmul(hiddenLayerTensor1, hiddenLayerWeightVariable2), hiddenLayerBiasVariable2))

outputLayerWeightVariable = tf.Variable(tf.random_normal([hiddenNodeCount2, outputNodeCount]))
outputLayerBiasVariable   = tf.Variable(tf.random_normal([outputNodeCount]))
outputLayerTensor         = tf.matmul(hiddenLayerTensor2, outputLayerWeightVariable) + outputLayerBiasVariable

costTensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = outputLayerTensor, labels = outputValueTensor))
optimizerOperation = tf.train.AdamOptimizer(learningRate).minimize(costTensor)

averageList = []
epochList   = []

initializerOperation = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializerOperation)
    for epoch in range(trainingEpochCount):
        averageCost = 0.
        totalBatch = int(numberPixelDatasets.train.num_examples / batchSize)
        for i in range(totalBatch):
            xBatchNDArray, yBatchNDArray = numberPixelDatasets.train.next_batch(batchSize)
            sess.run(optimizerOperation, feed_dict = {inputValueTensor : xBatchNDArray, outputValueTensor : yBatchNDArray})
            averageCost += sess.run(costTensor, feed_dict = {inputValueTensor : xBatchNDArray,\
                outputValueTensor : yBatchNDArray}) / totalBatch
        if epoch % displayStep == 0:
            print("회차 :", '%04d' % (epoch + 1), "비용 =", "{:.9f}".format(averageCost))
        averageList.append(averageCost)
        epochList.append(epoch + 1)
    print("훈련 단계 완료")

    correctPredictionTensor = tf.equal(tf.argmax(outputLayerTensor, 1), tf.argmax(outputValueTensor, 1))
    accuracyTensor          = tf.reduce_mean(tf.cast(correctPredictionTensor, "float"))

    print("모델 정확도 :", accuracyTensor.eval({inputValueTensor : numberPixelDatasets.test.images,\
        outputValueTensor : numberPixelDatasets.test.labels}))

    pp.plot(epochList, averageList, 'o', label = "MLP Training phase")
    pp.ylabel("cost")
    pp.xlabel("epoch")
    pp.legend()
    pp.show()