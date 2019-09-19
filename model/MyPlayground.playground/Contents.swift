import CreateML
import Foundation

let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/ahmedadel/Desktop/better-rest.json"))
let (trainingData,testingData) = data.randomSplit(by: 0.8) // 80% for training and 20% for testing

let regressor = try MLRegressor(trainingData: trainingData, targetColumn: "actualSleep")

let evaluationMetrics = regressor.evaluation(on: testingData)
print(evaluationMetrics.rootMeanSquaredError)
print(evaluationMetrics.maximumError)

let metaData = MLModelMetadata(author: "Ahmed Adel", shortDescription: "Model trained to predict sleep times for coffee drinkers", version: "1.0")

try regressor.write(to: (URL(fileURLWithPath: "/Users/ahmedadel/Desktop/sleepCalculator.mlmodel")), metadata: metaData)

