#机器学习特征选择
#代码来源
#https://developer.aliyun.com/article/655589
set.seed(1234)
library(mlbench)
library(caret)

# install.packages("mlbench")
# install.packages("caret")
# install.packages("tidyselect")

data(PimaIndiansDiabetes)
Matrix <- PimaIndiansDiabetes[,1:8]

library(Hmisc)
up_CorMatrix <- function(cor,p) {ut <- upper.tri(cor) 
data.frame(row = rownames(cor)[row(cor)[ut]] ,
           column = rownames(cor)[col(cor)[ut]], 
           cor =(cor)[ut] ) }

res <- rcorr(as.matrix(Matrix))
cor_data <- up_CorMatrix (res$r)
cor_data <- subset(cor_data, cor_data$cor > 0.5)
cor_data


data(PimaIndiansDiabetes)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


#递归特征消除
# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
data(PimaIndiansDiabetes)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(PimaIndiansDiabetes[,1:8], PimaIndiansDiabetes[,9], sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
