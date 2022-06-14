library(Minirand)

ntrt <- 3  #分组数
nsample <- 120  # 样本数
trtseq <- c(1, 2, 3) #分组序列
ratio <- c(2, 2, 1)  #分组比例


c1 <- sample(seq(1, 0), nsample, replace = TRUE, prob = c(0.4, 0.6)) 
c2 <- sample(seq(1, 0), nsample, replace = TRUE, prob = c(0.3, 0.7))
c3 <- sample(c(2, 1, 0), nsample, replace = TRUE, prob = c(0.33, 0.2, 0.5)) 
c4 <- sample(seq(1, 0), nsample, replace = TRUE, prob = c(0.33, 0.67)) 
# 生成受试者的协变量因子矩阵 
covmat <- cbind(c1, c2, c3, c4) # generate the matrix of covariate factors for the subjects
# label of the covariates  协变量的标记
colnames(covmat) = c("Gender", "Age", "Hypertension", "Use of Antibiotics") 
covwt <- c(1/4, 1/4, 1/4, 1/4) #equal weights  协变量权重

res <- rep(100, nsample) # result is the treatment needed from minimization method

#gernerate treatment assignment for the 1st subject  第一个受试者
res[1] = sample(trtseq, 1, replace = TRUE, prob = ratio/sum(ratio)) 


for (j in 2:nsample)
{
  # get treatment assignment sequentiall for all subjects 对所有受试者分配
  res[j] <- Minirand(covmat=covmat, j, covwt=covwt, ratio=ratio, 
                     ntrt=ntrt, trtseq=trtseq, method="Range", result=res, p = 0.9)
}
trt1 <- res
#Display the number of randomized subjects at covariate factors 显示协变量因素下随机受试者的数量 
balance1 <- randbalance(trt1, covmat, ntrt, trtseq) 
balance1
# 通过最小化算法计算总的不平衡度
totimbal(trt = trt1, covmat = covmat, covwt = covwt, 
         ratio = ratio, ntrt = ntrt, trtseq = trtseq, method = "Range")

covmat <- transform(covmat,group=trt1)

