#< Optimization >

# Mid-term
# Final update: 2021-04-25

# Method1: Elastic net(Ridge/Lasso regression) via coordinate decent
# Method2: Elastic logit(Ridge/Lasso logistic regression) via coordinate decent

#< Package >=================================================================================
library(magrittr)

#< Data >====================================================================================
train_data <- read.csv('C:\\Users\\Taner\\Desktop\\Optimization\\Data\\train.csv')
y <- ifelse(train_data[['classe']] == "b'A'",1,0)
x <- train_data[,1:121]
test_data <- read.csv('C:\\Users\\Taner\\Desktop\\Optimization\\Data\\test.csv')
test_y <- ifelse(test_data[['classe']] == "b'A'",1,0)
test_x <- test_data[,1:121]
#< Common function >=========================================================================
# Normalize
variable_norm <- function(x){return( (x - mean(x)) / sd(x) )}
# Sigmoid
sigmoid <- function(x,par_hat){
  lin_comb = (-1) * x %*% par_hat
  return(1 / (1 + exp(lin_comb)))
}
# Repeat experiment
repeat_experi <- function(x,y,FUN,experi_limit=50,repeat_cr=3,...){
  result_list <- vector(mode = 'list')
  stop_cr <- 0
  repeat{
    stop_cr = stop_cr + 1
    result_list <- rlist::list.append(.data = result_list,round(FUN(X,y,...),digits = 6))
    if (stop_cr > experi_limit){
      cat('Over',stop_cr,'time');break
    }else if (sum(duplicated(result_list)) > repeat_cr){
      cat('Number of experiments: ',stop_cr,'\n');break
    }
  }
  return(unique(result_list[duplicated(result_list)]))
}
# Calculate AIC and BIC
get_abic <- function(beta_coef,x,y,dist='gaussian'){
  X = apply(X = x,MARGIN = 2,variable_norm)
  X = cbind(1, X)
  n_obs = nrow(X)
  if (dist == 'gaussian'){
    pred = X %*% beta_coef
    rss = sum((pred - y)**2)
    k = sum(beta_coef != 0)
    model_aic = 2*k + n_obs*log(rss/n_obs)
    model_bic = (k+1)*log(n_obs) + n_obs*log(rss/n_obs)
  }else if (dist == 'binomial'){
    pred = sigmoid(X,beta_coef)
    logliklihood = sum(log(pred[y == 1]))+sum(log(1 - pred[y == 0]))
    k = sum(beta_coef != 0)
    model_aic = 2*k - 2*logliklihood
    model_bic = k*log(n_obs) - 2*logliklihood
  }
  return(list('AIC'=model_aic,'BIC'=model_bic))
}
#< Method1:function >========================================================================
elastic_net <- function(X,y,labda=0.01,alpha=1,epsilon=1e-4,init_par=NULL,max_iter=100,return_history=F){
  X = apply(X, MARGIN = 2,FUN = variable_norm)
  X = cbind(1, X)
  if (is.null(init_par)){
    init_par = rep(0,ncol(X))
  }else if(length(init_par) != ncol(X)){
      print("init_par's length is not correct");break
  }
  sample_size = length(y)
  gama = labda * alpha
  beta_hat = init_par
  update_beta = 1
  update_history = vector(mode = 'list')
  
  loop_num = 0
  while(norm(matrix(update_beta - beta_hat)) > epsilon){
    update_beta = beta_hat
    loop_num = loop_num + 1
    if (loop_num > max_iter){cat('Over',loop_num,' interation','\n') ; break}
    for (par_index in 1:length(init_par)){
      if (par_index == 1){
        beta_hat[par_index] = mean(y - X[,-par_index] %*% beta_hat[-par_index])
      }else{
        part1 = X[, par_index] %*% y / sample_size
        part2 = (apply(X, MARGIN = 2, FUN = `%*%`, X[,par_index]) %*% beta_hat) / sample_size
        z = part1[1] - part2[1] + beta_hat[par_index]
        threshold_value = ifelse(gama >= abs(z), 0, z + ifelse(z > 0, - gama, gama))
        beta_hat[par_index] = threshold_value / (1 + labda - gama)
      }
    }
    if (loop_num %% 5 == 0){print(paste0('Interation : ',loop_num))}
    
    if (return_history){
      update_history = rlist::list.append(update_history,beta_hat)
    }
  }
  print(paste0('Interation : ',loop_num))
  if (return_history){
    return(update_history)
  }else{
    beta_hat %<>% `names<-`(paste0('beta',1:length(beta_hat)))
    return(beta_hat)
  }
}
#< Method1 >=================================================================================
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

# Compare to package result
pkg_result <- glmnet::glmnet(x = model.matrix(y ~ .,data = data.frame(y,apply(X,2,variable_norm))),
                             y = y,family = 'gaussian',lambda = 0.1,intercept = T,alpha = 0.5)
pkg_result[['beta']]
#< Method2:function >========================================================================
elastic_logit <- function(X,y,labda=0.01,alpha=1,epsilon=1e-9,init_par=NULL,max_iter=50,return_history=FALSE){
  X = apply(X, MARGIN = 2,FUN = variable_norm)
  X = cbind(1, X)
  n_var = ncol(X)
  n_obs = nrow(X)
  
  if (is.null(init_par)){
    init_par = rep(0,n_var)
  }else if(length(init_par) != ncol(X)){
    print("init_par's length is not correct");break
  }
  
  gama = labda * alpha
  update_beta = beta_hat = init_par
  update_beta = update_beta + 1
  update_history = vector(mode = 'list')
  loop_num = 0
  
  while(norm(matrix(update_beta - beta_hat)) > epsilon){
    update_beta = beta_hat
    loop_num = loop_num + 1
    if (loop_num > max_iter){cat('Over',loop_num,' iteration','\n') ; break}
    
    px = sigmoid(X, beta_hat)
    px = ifelse(1 - px < 1e-5, 1,ifelse(px < 1e-5, 0, px))
    px_weight = ifelse(px %in% c(0,1),1e-10,px * (1 - px))
    z =( X %*% beta_hat + ((y - px)/px_weight))
    W = (t(px_weight) %*% X**2)[1,] / n_obs
    
    # all_y_hat <- sweep(x = X,MARGIN = 2,beta_hat,FUN = `*`)
    # update_f <- function(index){
    #   y_hat = rowSums(all_y_hat[,-index])
    #   update_value = (t(px_weight) %*% ((z - y_hat)*X[,index]))[1] / n_obs
    #   return(update_value)
    # }
    # 
    # all_update_value = unlist(lapply(X = seq(n_var),FUN = update_f))
    # beta_hat[1] = all_update_value[1] / W[1]
    # all_threshold_value = ifelse(gama >= abs(all_update_value), 0, all_update_value + ifelse(all_update_value > 0, - gama, gama))
    # beta_hat[-1] = all_threshold_value[-1] / (labda - gama +  W[-1])
    
    for (par_index in 1:length(init_par)){
      w = W[par_index]
      y_hat = X[,-par_index] %*% beta_hat[-par_index]
      update_value = (t(px_weight) %*% ((z - y_hat)*X[,par_index]))[1] / n_obs

      if (par_index == 1){
        beta_hat[par_index] = update_value / w
      }else{
        threshold_value = ifelse(gama >= abs(update_value), 0, update_value + ifelse(update_value > 0, - gama, gama))
        beta_hat[par_index] = threshold_value / (labda - gama +  w)
      }
    }
    beta_hat = round(beta_hat,6)
    if (return_history){
      update_history = rlist::list.append(update_history,beta_hat)
    }
    
    if (loop_num %% 5 == 0){print(paste0('Iteration : ',loop_num))}
  }
  print(paste0('Iteration : ',loop_num))
  beta_hat %<>% `names<-`(paste0('beta',seq(0,length(beta_hat)-1)))
  if (return_history){
    return(update_history)
  }else{
    return(beta_hat)
  }
}
#< Method2 >=================================================================================
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





pkg_result <- glmnet::glmnet(x = model.matrix(y ~ .,data = data.frame(y,apply(x,2,variable_norm))),
                             y = y,family = 'binomial',lambda = 0.004,intercept = T)
pkg_result[['beta']]
