##Set working directory

setwd("C://Users//Ashishkumar Rana//Desktop//MSC_Cariculum//SEM1//Case Study")
getwd()

# Reading the Data

library(readxl)
heart <- read.csv("heart.csv")
View(heart)

# Loading the required library
install.packages("corrplot")
install.packages("tidyverse")
install.packages("ggthemes")
install.packages("Hmisc")
library(tidyverse) 
library(ggthemes) 
library(ggplot2)
library(corrplot)
library(MASS)
library(Hmisc)


# Giving better names to categorical variables

heart$Sex[heart$Sex == 'F'] <- 0
heart$Sex[heart$Sex == 'M'] <- 1

heart$ChestPainType[heart$ChestPainType == 'ATA'] <- 2
heart$ChestPainType[heart$ChestPainType == 'NAP'] <- 3
heart$ChestPainType[heart$ChestPainType == 'ASY'] <- 4
heart$ChestPainType[heart$ChestPainType == 'TA']  <- 1

heart$ExerciseAngina[heart$ExerciseAngina == 'N'] <- 0
heart$ExerciseAngina[heart$ExerciseAngina == 'Y'] <- 1

heart$RestingECG[heart$RestingECG == 'Normal'] <- 0
heart$RestingECG[heart$RestingECG == 'ST']     <- 1
heart$RestingECG[heart$RestingECG == 'LVH']    <- 2

heart$ST_Slope[heart$ST_Slope == 'Up'] <- 1
heart$ST_Slope[heart$ST_Slope == 'Flat'] <- 2
heart$ST_Slope[heart$ST_Slope == 'Down'] <- 3

# Converting character type to numeric
heart$Age = as.numeric(heart$Age)
heart$Sex = as.numeric(heart$Sex)
heart$ChestPainType = as.numeric(heart$ChestPainType)
heart$RestingBP = as.numeric(heart$RestingBP)
heart$Cholesterol = as.numeric(heart$Cholesterol)
heart$FastingBS = as.numeric(heart$FastingBS)
heart$RestingECG = as.numeric(heart$RestingECG)
heart$MaxHR = as.numeric(heart$MaxHR)
heart$ExerciseAngina = as.numeric(heart$ExerciseAngina)
heart$ST_Slope = as.numeric(heart$ST_Slope)
heart$HeartDisease = as.numeric(heart$HeartDisease)

View(heart)
write.csv(heart,"C://Users//Ashishkumar Rana//Desktop//MSC_Cariculum//SEM1//Case Study//exportdata.csv")

# Correlation
heart_corr = cor(heart)
corrplot(heart_corr,method = 'number',
         type = 'lower',bg = 'black', addgrid.col = 'white', tl.col = 'black')

# Data description
#Barplot: Sex & Heart Disease

counts <- table(heart$HeartDisease, heart$Sex)
barplot((counts), main = "Heart Disease by Sex category", xlab = "Sex",
        col = c("green","red"), 
        legend = c("No Heart Disease","Heart Disease"),
        beside = TRUE,ylim = c(0,570))

#Age Distribution
ggplot(heart,aes(x = Age)) + geom_histogram(bins = 30, fill ="dodgerblue4") +
  ggtitle("Age Distribution") +ylab("number of people") + ylim(c(0,80))

#Analysis of Chest Pain
cp = table(heart$HeartDisease,heart$ChestPainType)
barplot(cp, beside = T, main = "Heart Disease by Chest Pain", 
        xlab = "Chest Pain Type", ylab = "Frequency", ylim = c(0,500),
        col = c("green","red"), legend = c("No Heart Disease", "Heart Disease"))

#Data distribution in Dataset for each distribution
hist.data.frame(heart)

#Scatter Plot: Age vs Resting BP vs Heart Disease

ggplot(heart, aes(x=Age, y=RestingBP, color = HeartDisease)) + 
  geom_point(size = 3) + ylim(c(0,250)) + 
  ggtitle("Scatter Plot: Age vs Resting BP vs Heart Disease")

#Histogram of Age Distribution

library(ggplot2)

ggplot(heart, aes(Age)) +
  geom_histogram( fill="yellow",
                  binwidth = 5,
                  col="red",
                  size=.5) +  # change binwidth
  labs(title="Age Distribution", 
       x="Age",
       y="Count")+theme(plot.title = element_text(hjust = 0.5,size=20))


#Frequency Distribution of sex 

g <- ggplot(heart, aes(Sex))
g + geom_bar(fill="plum") +
  labs(title="Bar Chart", 
       subtitle="Frequency distribution of Sex",
       x="Sex",
       y="Frequency")

#Scatterplot (Age Vs Cholesterol)

gg <- ggplot(heart, aes(x=Age, y=Cholesterol))

gg+geom_point() + 
  labs(subtitle="Age Vs Cholesterol", 
       y="Cholesterol", 
       x="Age",  title="Scatterplot")

#Scatterplot (Age Vs MaxHR)

gg <- ggplot(heart, aes(x=Age, y=MaxHR))
gg+geom_point() + 
  labs(subtitle="Age Vs MaxHR", 
       y="MaxHR", 
       x="Age",  title="Scatterplot")