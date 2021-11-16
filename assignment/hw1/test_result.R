#< Optimization >

# Mid-term
# Final update: 2021-04-25

# Method1: Elastic net(Ridge/Lasso regression) via coordinate decent
# Method2: Elastic logit(Ridge/Lasso logistic regression) via coordinate decent

#< Package/Function >=================================================================================
library(magrittr)
source("./utils.R")
source("./elastic_net.R")
source("./elastic_logit.R")

#< Data >=============================================================================================
train_data <- read.csv('./Data/train.csv')
y <- ifelse(train_data[['classe']] == "b'A'",1,0)
x <- train_data[,1:121]
test_data <- read.csv('./Data/test.csv')
test_y <- ifelse(test_data[['classe']] == "b'A'",1,0)
test_x <- test_data[,1:121]

#< Method1 >==========================================================================================
# Chose lambda
temp_result = rep(0,ncol(x)+1)
all_beta_coef = all_abic = vector(mode = 'list')
a = seq(0.005,0.009,0.001)
b = seq(0.010,0.015,0.001)
for (lbd in b){
  temp_result = elastic_net(x,(2*y)-1,labda = lbd,init_par = temp_result,epsilon = 0.0001)
  all_beta_coef = rlist::list.append(all_beta_coef,temp_result)
  abic = get_abic(beta_coef = temp_result,x = x,y = (2*y)-1,dist = 'gaussian')
  all_abic = rlist::list.append(all_abic,abic)
  print(lbd)
  print(abic)
}


final_beta = elastic_net(x,2*y-1,labda = 0.02,alpha = 1,init_par = temp_result,epsilon = 0.0001)
lin_pred = ifelse(cbind(1,apply(X = test_x,MARGIN = 2,FUN = variable_norm)) %*% final_beta > 0,1,-1)
sum(diag(table(lin_pred,test_y))) / sum(table(lin_pred,test_y))

# Plot
beta_history <- elastic_net(x,2*y-1,labda = 0.02,alpha = 1,epsilon = 0.0001,return_history = T)
beta_history <- do.call(rbind,beta_history)
beta_history <- rbind(rep(0,dim(beta_history)[2]),beta_history)
matplot(beta_history,pch=1,type='l',col=4,xlab = 'Iteration',ylab = 'Coefficient',lty=1,main='Linear regression coefficient')

# Compare to package's result
pkg_result <- glmnet::glmnet(x = model.matrix(y ~ .,data = data.frame(y,apply(X,2,variable_norm))),
                             y = y,family = 'gaussian',lambda = 0.1,intercept = T,alpha = 0.5)
pkg_result[['beta']]

#< Method2 >==========================================================================================
# Chose lambda
temp_result = rep(0,ncol(x)+1)
all_beta_coef = all_abic = vector(mode = 'list')
for (lbd in seq(0.003,0.005,0.0005)){
  temp_result = elastic_logit(x,y,labda = lbd,init_par = temp_result,max_iter = 100,epsilon = 0.001,return_history = F)
  all_beta_coef = rlist::list.append(all_beta_coef,temp_result)
  abic = get_abic(beta_coef = temp_result,x = x,y = y,dist = 'binomial')
  all_abic = rlist::list.append(all_abic,abic)
  print(lbd)
  print(abic)
}

final_beta = elastic_logit(x,y,labda = 0.04,alpha = 1,epsilon = 0.0001,return_history = T)
pred_prob = sigmoid(x = cbind(1,apply(X = test_x,MARGIN = 2,FUN = variable_norm)),par_hat = final_beta[[length(final_beta)]])

pred_c <- ifelse(pred_prob>0.5,1,0)
sum(diag(table(pred_c,test_y))) / sum(table(pred_c,test_y))
cvAUC::AUC(predictions = pred_c,labels = test_y)
pROC::auc(test_y, pred_c[,1])
pROC::roc(test_y, pred_c[,1],plot=T)

final_beta[[21]][final_beta[[21]] != 0]
which(final_beta[[21]] != 0)

test_result <- do.call(rbind,final_beta)
test_result <- rbind(rep(0,dim(test_result)[2]),test_result)
matplot(test_result,pch=1,type='l',col=4,xlab = 'Iteration',ylab = 'Coefficient',lty=1,main='Logistic regression coefficient')


# Compare to package's result
pkg_result <- glmnet::glmnet(x = model.matrix(y ~ .,data = data.frame(y,apply(x,2,variable_norm))),
                             y = y,family = 'binomial',lambda = 0.004,intercept = T)
pkg_result[['beta']]