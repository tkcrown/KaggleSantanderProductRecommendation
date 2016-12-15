rm(list = ls())
setwd("~/Documents/kaggle/santar/code_xz/")
library(data.table)
load("../data/june_added.RData")
# read in train
train = fread('../input/train_ver2.csv')
test = fread("../input/test_ver2.csv")

result_dt <- test
result_dt$fecha_dato <- NULL

train <- train[fecha_dato %in% c('2016-01-28', '2016-02-28',
                                 '2016-03-28', '2016-04-28',
                                 '2016-05-28')]

raw_feature_extract <- function(train, date, result_dt){
  data <- train[fecha_dato == date]
  data$feacha_dato <- NULL
  selected_fea <- c('ncodpers', 'age', 'antiguedad', 'indrel', 'indrel_1mes', 
                    'tipodom', 'cod_prov', 'ind_actividad_cliente', 'renta',
                    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                    'ind_nomina_ult1',   'ind_nom_pens_ult1', 'ind_recibo_ult1')
  data <- data[, selected_fea, with = F]
  colnames(data)[2:dim(data)[2]] <- paste(date, selected_fea[2:dim(data)[2]], sep = "__")
  result <- merge(result_dt, data, by = 'ncodpers', all.x = TRUE)
  return(result)
}

result <- result_dt
for(date in c('2016-01-28', '2016-02-28','2016-03-28', '2016-04-28','2016-05-28')){
  result <- raw_feature_extract(train, date, result)
}

# convert character to integer
cat <- colnames(result)[which(sapply(result, class) == "character")]

result[,(cat) := lapply(.SD, as.factor), .SDcols = cat]
result[,(cat) := lapply(.SD, as.numeric), .SDcols = cat]

write.csv(result, file = "../data/test201506_v0.csv", row.names = F)

