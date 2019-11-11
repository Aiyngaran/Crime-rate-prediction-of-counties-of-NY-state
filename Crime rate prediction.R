data <- read.csv("D:/ADA/project/New folder/final_1.csv")
data <- data[,-c(1,2,4,7,8,10,12,14,17,19,21,23,25,27,29,30,32,35,36,38,40,42,44)]
data <- na.omit(data)
data$Crime.count.per.10000 <- (log(data$Crime.count.per.10000+1))

plot(density(data$Crime.count.per.10000))

library(corrplot)
cor_matrix <- cor(data[,-2])

corrplot(cor_matrix)

#####Random forest and Bagging

library(randomForest)
library(caret)
set.seed(12343)
smp_size <- floor(0.75 * nrow(data))
train_index <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_index, ]
test <- data[-train_index, ]

options(java.parameters = "-Xms5g") # or whatever g u wanna set to
library(RJDBC)
library (rJava)
library (bartMachine)
set_bart_machine_num_cores(4)

library(Metrics)
library(glmnet)
set.seed(1989)
data_1 <- model.matrix(~., data = data)
train_ind <- sample(1:nrow(data_1), 0.75 * nrow(data_1))
test_ind <- setdiff(1:nrow(data_1), train_ind)
x_train <- data_1[train_ind, 3:ncol(data_1)]
y_train <- data_1[train_ind, 2 ]
x_test <- data_1[test_ind, 3:ncol(data_1)]
y_test <- data_1[test_ind, 2]

cv <- cv.glmnet(x_train , y_train, alpha = 1)
minlamda <- cv$lambda.min

RF1 <- randomForest(Crime.count.per.10000~., data = train,mtry = 7, ntree = 400,importance = TRUE)
RF2 <- randomForest(Crime.count.per.10000~., data = train,mtry = 4, ntree = 400,importance = TRUE)
RF3 <- randomForest(Crime.count.per.10000~., data = train,mtry = 5, ntree = 400,importance = TRUE)
RF4 <- randomForest(Crime.count.per.10000~., data = train,mtry = 21,ntree = 400,importance = TRUE)

which.min(plot(RF1))
which.min(plot(RF2))
which.min(plot(RF3))
which.min(plot(RF4))

plot(RF1, main = '')
points(385, RF1$mse[385])
par(new= TRUE)
plot(RF2, main = '', col = 'green')
points(220, RF2$mse[220])
par(new= TRUE)
plot(RF3, main = '', col = 'red')
points(198, RF3$mse[198])
par(new = TRUE)
plot(RF4,main = '', col = 'blue')
points(300,RF4$mse[300])

RF.pred_7 <- predict(RF1,test)
RMSE(RF.pred_7,test$Crime.count.per.10000)

RF.pred_4 <- predict(RF2,test)
RMSE(RF.pred_4,test$Crime.count.per.10000)

RF.pred_5 <- predict(RF3,test)
RMSE(RF.pred_5,test$Crime.count.per.10000)

RF.pred_21 <- predict(RF4,test)
RMSE(RF.pred_21,test$Crime.count.per.10000)

# RF with mtry = 7 gives the lowest test RMSE. Hence it is chosen among the 4 RF models

################ Boosting ##############

library(caret)
library(gbm)
library(Metrics)
caretGrid <- expand.grid(interaction.depth=c(1,2, 3,4, 5,6), n.trees = (0:100)*50,
                         shrinkage=c(0.01, 0.001),
                         n.minobsinnode=10)
metric <- "RMSE"
trainControl <- trainControl(method="cv", number=5)

set.seed(99)
gbm.caret <- train(Crime.count.per.10000~ ., data=train, distribution="gaussian", method="gbm",
                   trControl=trainControl, verbose=FALSE, 
                   tuneGrid=caretGrid, metric=metric)                  
caret.predict <- predict(gbm.caret, newdata=test,n.trees = 5000)

rmse.caret<-rmse(test$Crime.count.per.10000, caret.predict)
print(rmse.caret)

R2.caret <- cor(gbm.caret$finalModel$fit, train$Crime.count.per.10000)^2
print(R2.caret)
gbm.caret$bestTune

########### MARS ############
library(Metrics)
library(earth)

# Runnning with backward selection

train.error.mars <- array(numeric(),c(3,8)) 
test.error.mars <- array(numeric(),c(3,8)) 
for (i in 1:3){
  for (j in 1:8){
    model.mars <- earth(train$Crime.count.per.10000~.,
                        data=train, degree=i, penalty=j, pmethod="backward", nfold=10, ncross=5)
    pred.train.mars <- predict(model.mars,train)
    pred.test.mars <- predict(model.mars,test)
    train.error.mars[i,j] <- RMSE(train$Crime.count.per.10000,pred.train.mars)
    test.error.mars[i,j] <- RMSE(test$Crime.count.per.10000,pred.test.mars)
  }
}
train.error.mars
test.error.mars
min(test.error.mars)


# RUnning with forward selection

train.error.mars1 <- array(numeric(),c(3,8)) 
test.error.mars1 <- array(numeric(),c(3,8)) 
for (i in 1:3){
  for (j in 1:8){
    model.mars <- earth(train$Crime.count.per.10000~.,
                        data=train, degree=i, penalty=j, pmethod="forward", nfold=10, ncross=5)
    pred.train.mars <- predict(model.mars,train)
    pred.test.mars <- predict(model.mars,test)
    train.error.mars1[i,j] <- RMSE(train$Crime.count.per.10000,pred.train.mars)
    test.error.mars1[i,j] <- RMSE(test$Crime.count.per.10000,pred.test.mars)
  }
}
train.error.mars1
test.error.mars1

# not much difference between forward and backward selection
# Running final mars model with degree = 2, penalty = 7 and with backward selection

###### Running CV for all models ###############

library(randomForest)
library(gbm)
library(earth)
library(mgcv)
library(caret)
set.seed(10000)
numholdout = 30
percentholdout = 0.2
nmodel = 16
vecRMSE = matrix(data = NA, ncol = nmodel, nrow = numholdout)
vecMAE = matrix(data = NA, ncol = nmodel, nrow = numholdout)
randomstring <- function(percent,length){
  s <- c()
  for(j in 1:length){
    if(runif(1) <= percent){
      s[j] <- 1
    }
    else{
      s[j] <- 0
    }
  }
  s
}

attach(data)
for(i in 1:numholdout){
  s <- randomstring(percentholdout, nrow(data))
  tmp.data <- cbind(data,s)
  tmp.response <- cbind(data$Crime.count.per.10000, s)
  holdout <- subset(tmp.data, s==1)[,2:length(data)]
  
  holdout_matrix <- model.matrix(~., data = holdout)
  x_test <- holdout_matrix[, 2: ncol(holdout)]
  
  holdout.response <- subset(tmp.response,s==1)[,1]
  
  y_test <- holdout.response
  
  train <- subset(tmp.data, s==0)[,1:length(data)]
  
  features_train_x <- train[, -1]
  features_train_y <- train[, 1]
  
  train_1 <- model.matrix(~., data = train)
  x_train <- train_1[, 3: ncol(train)]
  y_train <- train_1[, "Crime.count.per.10000"]
  
  sizeholdout <- dim(holdout)[1]
  sizetrain <- dim(train)[1]
  
  # BART Model
  
  bart_model <- bartMachine(features_train_x, features_train_y,q=0.99,k=3,nu=3,num_trees = 100, mem_cache_for_speed=FALSE)
  bart_pred_train <- predict(bart_model,features_train_x)
  bart_pred_test <- predict(bart_model,holdout)
  
  # Random forest
  randomforest <- randomForest(Crime.count.per.10000 ~.,data = train, mtry = 7, ntree = 400, importance =TRUE)
  rf_train_predict <-predict(randomforest, newdata=train,type='response')
  rf_test_predict <- predict(randomforest, newdata=holdout,type='response')
  
  # GBM
  gbm <- gbm(Crime.count.per.10000~.,data=train,shrinkage = 0.01,interaction.depth = 6,n.trees = 5000,n.minobsinnode = 10)
  pred_train_gbm <- predict(gbm,newdata=train,n.trees = 5000)
  pred_test_gbm <- predict(gbm, newdata = holdout,n.trees = 5000)
  
  # MARS
  mars <- earth(Crime.count.per.10000~.,data=train,degree = 2,penalty = 7,pmethod = "backward")
  pred_train_mars <- predict(mars,train)
  pred_test_mars <- predict(mars,holdout)
  
  # GAM
  gam <- gam(formula = (Crime.count.per.10000 ~ s(WA.female.per.10000,bs="cr",by=Crime.type)+Crime.type+s(Arrest.count.per.10000,bs="cr")+s(H.male.per.10000,bs="cr",by=Crime.type)+s(Average.of.Unemployment.Rate.per.10000,bs="cr",by=Crime.type)),family=gaussian,train,method="REML")
  pred_train_gam <- predict(gam,train)
  pred_test_gam <- predict(gam,holdout)
  
  #ridge
  ridge <- glmnet(x_train, y_train, alpha = 1, lambda = minlamda)
  ridge_pred_train <- predict(ridge, s = minlamda, newx = x_train)
  ridge_pred_test <- predict(ridge, s = minlamda, newx = x_test)
  
  #lasso
  lasso <- glmnet(x_train, y_train, alpha = 0, lambda = minlamda)
  lasso_pred_train <- predict(lasso, s = minlamda, newx = x_train)
  lasso_pred_test <- predict(lasso, s = minlamda, newx = x_test)
  
  
  # NULL
  null.predict <- mean(train$Crime.count.per.10000)

  # calculate train and test MSE and MAE for RF
  vecRMSE[i,1]<-sqrt(mean((train$Crime.count.per.10000-rf_train_predict)^2)) # train RMSE
  vecMAE[i,1]<-mean(abs(train$Crime.count.per.10000-rf_train_predict)) # train MAE
  
  vecRMSE[i,2] <- sqrt(mean((holdout.response-rf_test_predict)^2)) # test RMSE
  vecMAE[i,2]<-mean(abs(holdout.response-rf_test_predict)) # test MAE
  
  # calculate MSE and MAE for model GBM
  vecRMSE[i,3]<-sqrt(mean((train$Crime.count.per.10000-pred_train_gbm)^2))
  vecMAE[i,3]<-mean(abs(train$Crime.count.per.10000-pred_train_gbm))
  
  vecRMSE[i,4] <- sqrt(mean((holdout.response-pred_test_gbm)^2))
  vecMAE[i,4]<-mean(abs(holdout.response-pred_test_gbm))
  
  # calculate MSE and MAE for model MARS
  vecRMSE[i,5]<-sqrt(mean((train$Crime.count.per.10000-pred_train_mars)^2))
  vecMAE[i,5]<-mean(abs(train$Crime.count.per.10000-pred_train_mars))
  
  vecRMSE[i,6] <- sqrt(mean((holdout.response-pred_test_mars)^2))
  vecMAE[i,6]<-mean(abs(holdout.response-pred_test_mars))
  
  # Calculate MSE and MAE for model GAM
  vecRMSE[i,7]<-sqrt(mean((train$Crime.count.per.10000-pred_train_gam)^2))
  vecMAE[i,7]<-mean(abs(train$Crime.count.per.10000-pred_train_gam))
  
  vecRMSE[i,8] <- sqrt(mean((holdout.response-pred_test_gam)^2))
  vecMAE[i,8]<-mean(abs(holdout.response-pred_test_gam))
  
  
  # Calculate MSE and MAE for NULL model
  vecRMSE[i,9] <- sqrt(mean((train$Crime.count.per.10000-null.predict)^2))
  vecMAE[i,9] <- mean(abs(train$Crime.count.per.10000-null.predict))
  
  vecRMSE[i,10] <- sqrt(mean((holdout.response-null.predict)^2))
  vecMAE[i,10] <- mean(abs(holdout.response-null.predict)^2)
  
  # calculate MSE and MAE for model ridge
  
  vecRMSE[i,11] <- sqrt(mean((holdout.response-ridge_pred_train)^2)) # test RMSE
  vecMAE[i,11]<-mean(abs(holdout.response-ridge_pred_train)) # test MAE
  
  vecRMSE[i,12] <- sqrt(mean((holdout.response-ridge_pred_test)^2)) # test RMSE
  vecMAE[i,12]<-mean(abs(holdout.response-ridge_pred_test)) # test MAE
  
  # calculate MSE and MAE for model lasso
  
  vecRMSE[i,13] <- sqrt(mean((holdout.response-lasso_pred_train)^2)) # test RMSE
  vecMAE[i,13]<-mean(abs(holdout.response-lasso_pred_train)) # test MAE
  
  vecRMSE[i,14] <- sqrt(mean((holdout.response-lasso_pred_test)^2)) # test RMSE
  vecMAE[i,14]<-mean(abs(holdout.response-lasso_pred_test)) # test MAE
  
  vecRMSE[i,15]<-rmse(bart_pred_train, features_train_y)
  vecMAE[i,15]<-mean(abs(holdout.response-bart_pred_train))
  
  vecRMSE[i,16]<-rmse(bart_pred_test, holdout.response)
  vecMAE[i,16]<-mean(abs(holdout.response-bart_pred_test))
  
  
}

meanRMSE<-c()
for(k in 1:10){
  meanRMSE[k]<-mean(vecRMSE[,k])
}

meanMAE<-c()
for(k in 1:10){
  meanMAE[k]<- mean(vecMAE[,k])
}

meanRMSE
min(meanRMSE)
meanMAE
min(meanMAE)

########################## Model summaries ##################

summary(gam)
summary(mars)
summary(randomforest)
summary(gbm)

########################## QQ plots #################

library(car)
residuals.mars <- holdout.response - pred_test_mars
residuals.gam <- holdout.response - pred_test_gam
residuals.rf <- holdout.response - rf_test_predict
residuals.gbm <- holdout.response - pred_test_gbm

qqPlot(residuals.mars, main = "MARS MODEL: Residual Plot") 
qqPlot(residuals.gam, main = "GAM MODEL: Residual Plot") 
qqPlot(residuals.rf, main = "RF MODEL: Residual Plot") 
qqPlot(residuals.gbm, main = "GBM MODEL: Residual Plot") 

######################## Actual vs predicted ################

plot(holdout.response, pred_test_mars, pch="o", col='black',lty=5,  main="MARS: Actual vs. Predicted",
     xlab = "Actual crime rate", ylab="Predicted crime rate")
abline(0,1)

plot(holdout.response, pred_test_gam, pch="o", col='black',lty=5,  main="GBM: Actual vs. Predicted",
     xlab = "Actual crime rate", ylab="Predicted crime rate")
abline(0,1)

plot(holdout.response, rf_test_predict, pch="o", col='black',lty=5,  main="RF: Actual vs. Predicted",
     xlab = "Actual crime rate", ylab="Predicted crime rate")
abline(0,1)

plot(holdout.response, pred_test_gbm, pch="o", col='black',lty=5,  main="GBM: Actual vs. Predicted",
     xlab = "Actual crime rate", ylab="Predicted crime rate")
abline(0,1)


################### partial plots for GBM model #################

plot(gbm,i="Crime.type")
plot(gbm,i="Arrest.count.per.10000")
plot(gbm,i="total.male.per.10000")
plot(gbm,i="total.female.per.10000")
plot(gbm,i="WA.female.per.10000")
plot(gbm,i="BAC.female.per.10000")
plot(gbm,i="WA.male.per.10000")
plot(gbm,i="BA.female.per.10000")
plot(gbm,i="BAC.male.per.10000")
plot(gbm,i="Average.of.Unemployment.Rate.per.10000")
plot(gbm,i="prison.population.per.10000")
plot(gbm,i="firearm.count.per.10000")
plot(gbm,i="firearm.rate.per.10000")

library(pdp)

gbm_pdp <- partial(gbm,pred.var = "Crime.type",n.trees = 5000)

barplot(gbm_pdp$yhat,names.arg = gbm_pdp$Crime.type,cex.names = 0.5)