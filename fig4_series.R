library("ggpubr")
library(ggplot2)
# relief_result <- read.csv("~/GitHub/Rice_authenticity_ICP_new/updated_results/relief_result_python_with_constant.csv")
relief_result<-read.csv("~/GitHub/Rice_authenticity_ICP_new/updated_results/relief_result_python.csv")
tiff("~/GitHub/4a.tiff", res=300, height=115, width=95, units="mm")


ggbarplot(relief_result, x = "feature", y = "score",
          fill = "steelblue",               # change fill color by cyl
          color = "white",            # Set bar border colors to white
          # palette = "jco",            # jco journal color palett. see ?ggpar
          sort.val = "asc",          # Sort the value in dscending order
          sort.by.groups = FALSE,     # Don't sort inside each group
          x.text.angle = 0,          # Rotate vertically x axis texts
          rotate=TRUE,
          ggtheme = theme_pubr()   
) + theme(legend.position = "none", text = element_text(size = 6, face = "bold"))+labs(y="Relative importance", x = "Features")

dev.off()

training_results<- read.csv("~/GitHub/Rice_authenticity_ICP_new/updated_results/training_result.csv")
# training_results$Number.of.features<- as.factor(training_results$Number.of.features)
tiff("~/GitHub/4b.tiff", res=300, height=115, width=95, units="mm")
# ggplot(training_results, aes(x=Number.of.features, y=Accuracy, color=Algorithm,fill=Algorithm, shape = Algorithm)) + geom_line() +geom_point(size=4)
ggplot(training_results, aes(x=ï..Number.of.features, 
                             y=Accuracy, 
                             color=Algorithm,
                             fill=Algorithm, 
                             shape = Algorithm)
       ) +geom_line() +geom_point(size=4)+theme_pubr()+theme(text = element_text(size = 6, face = "bold"),
                                                             legend.position = "bottom"
                                                )+labs(y="Accuracy (%)", x = "Number.of.features")
dev.off()
