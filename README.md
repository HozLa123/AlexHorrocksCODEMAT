% Import dataset and turn to an array
opts = detectImportOptions('diabetes.xlsx');
data = readtable('diabetes.xlsx', opts);
data = table2array(data);
X = data(:,1:end-1);
y = data(:,end);

% Split the dataset to training set and a test set
cv = cvpartition(size(data,1),'HoldOut',0.3);
idx = cv.test;

% Separate to training and test data
XTrain = X(~idx,:);
yTrain = y(~idx,:);
XTest = X(idx,:);
yTest = y(idx,:);

% Train a logistic model
mdl_log = fitglm(XTrain, yTrain, 'Distribution', 'binomial');

% Results
yPred_log = predict(mdl_log, XTest);

% Adjust the threshold for classifying as positive
threshold = 0.4;
yPred_log = yPred_log > threshold;

% Convert yPred type
yPred_log = double(yPred_log);

% Confusion matrix
confusionMatrix_log = confusionmat(yTest, yPred_log);

% Display confusion matrix
disp('Confusion matrix for Logistic Regression:')
disp(confusionMatrix_log)

% Metrics for Logistic Regression
accuracy_log = sum(yPred_log == yTest) / length(yTest);
TP_log = confusionMatrix_log(2,2);
FP_log = confusionMatrix_log(1,2);
FN_log = confusionMatrix_log(2,1);
precision_log = TP_log / (TP_log + FP_log);
recall_log = TP_log / (TP_log + FN_log);
f1_log = 2 * (precision_log * recall_log) / (precision_log + recall_log);

% Display metrics
disp(['Accuracy for Logistic Regression: ', num2str(accuracy_log)])
disp(['Precision for Logistic Regression: ', num2str(precision_log)])
disp(['Recall for Logistic Regression: ', num2str(recall_log)])
disp(['F1 Score for Logistic Regression: ', num2str(f1_log)])

% Plot confusion matrix
figure;
confusionchart(yTest, yPred_log);
title('Confusion Matrix for Logistic Regression')

% Train a Linear Discriminant Analysis model
mdl_lda = fitcdiscr(XTrain, yTrain);

% Predict probabilities
[~,postProb_lda] = predict(mdl_lda,XTest);

% Apply the threshold to make predictions
yPred_lda = postProb_lda(:,2) > threshold;

% Convert yPred to numeric
yPred_lda = double(yPred_lda);

% Create confusion matrix
confusionMatrix_lda = confusionmat(yTest, yPred_lda);

% Display confusion matrix
disp('Confusion matrix for Linear Discriminant Analysis:')
disp(confusionMatrix_lda)

% Calculate performance metrics for Linear Discriminant Analysis
accuracy_lda = sum(yPred_lda == yTest) / length(yTest);
TP_lda = confusionMatrix_lda(2,2);
FP_lda = confusionMatrix_lda(1,2);
FN_lda = confusionMatrix_lda(2,1);
precision_lda = TP_lda / (TP_lda + FP_lda);
recall_lda = TP_lda / (TP_lda + FN_lda);
f1_lda = 2 * (precision_lda * recall_lda) / (precision_lda + recall_lda);

% Display performance metrics
disp(['Accuracy for Linear Discriminant Analysis: ', num2str(accuracy_lda)])
disp(['Precision for Linear Discriminant Analysis: ', num2str(precision_lda)])
disp(['Recall for Linear Discriminant Analysis: ', num2str(recall_lda)])
disp(['F1 Score for Linear Discriminant Analysis: ', num2str(f1_lda)])

% Plot confusion matrix
figure;
confusionchart(yTest, yPred_lda);
title('Confusion Matrix for Linear Discriminant Analysis')

% Create a bar chart to compare the performance of the two models
figure;
performance_metrics = [accuracy_log, precision_log, recall_log, f1_log; accuracy_lda, precision_lda, recall_lda, f1_lda];
bar(performance_metrics)
set(gca,'xticklabel',{'Logistic Regression','Linear Discriminant Analysis'})
legend('Accuracy', 'Precision', 'Recall', 'F1 Score')
title('Comparison of Performance Metrics for Two Models')
Confusion matrix for Logistic Regression:
   110    28
    33    59

Accuracy for Logistic Regression: 0.73478
Precision for Logistic Regression: 0.67816
Recall for Logistic Regression: 0.6413
F1 Score for Logistic Regression: 0.65922
Confusion matrix for Linear Discriminant Analysis:
   113    25
    35    57

Accuracy for Linear Discriminant Analysis: 0.73913
Precision for Linear Discriminant Analysis: 0.69512
Recall for Linear Discriminant Analysis: 0.61957
F1 Score for Linear Discriminant Analysis: 0.65517
>> 
