# Administrative Stuff
setwd("/Users/freeg007/Documents/Coursera/Johns\ Hopkins\ University/Data\ Science/Capstone\ Project")

# Does the link lead to an HTML page describing the exploratory analysis of the training data set?
# Goal --> 1: yes, the link does lead to a document describing the exploratory analysis

# Has the data scientist done basic summaries of the three files? Word counts, line counts and basic data tables?
# Goal --> 1: yes, the data scientist has evaluated basic summaries of the data such as word and line counts

# Has the data scientist made basic plots, such as histograms to illustrate features of the data?
# Goal -->yes, the data scientist has made basic plots, such as histograms to illustrate features of the data

# Was the report written in a brief, concise style, in a way that a non-data scientist manager could appreciate?
# Goal --> yes, the report could be understood by a non data scientist and is brief and concise


# Has the data scientist done basic summaries of the three files? Word counts, line counts and basic data tables?

install.packages("Rcpp", repos="http://cran.rstudio.com/")
install.packages("ggplot2", repos="http://cran.rstudio.com/") 
install.packages("R.utils", repos="http://cran.rstudio.com/")
install.packages("tm", repos="http://cran.rstudio.com/")
install.packages("RWeka", repos="http://cran.rstudio.com/")
install.packages("SnowballC", repos="http://cran.rstudio.com/")
install.packages("dplyr", repos="http://cran.rstudio.com/")
install.packages("wordcloud", repos="http://cran.rstudio.com/")
install.packages("RColorBrewer", repos="http://cran.rstudio.com/")
install.packages("grid", repos="http://cran.rstudio.com/")

library(ggplot2)
library(R.utils)
library(tm)
library(SnowballC)
library(RWeka)
library(ggplot2)
library(Rcpp)
library(dplyr)
library(wordcloud)
library(RColorBrewer)
library(grid)

# Has the data scientist done basic summaries of the three files? Word counts, line counts and basic data tables?

## DATA PROCESSING
englishBlogs <- readLines("./final/en_US/en_US.blogs.txt", encoding = "UTF-8", skipNul=TRUE)
englishNews <- readLines("./final/en_US/en_US.news.txt", encoding = "UTF-8", skipNul=TRUE)
englishTwitter <- readLines("./final/en_US/en_US.twitter.txt", encoding = "UTF-8", skipNul=TRUE)

SAMPLE_SIZE = 10000
sampleTwitter <- englishTwitter[sample(1:length(englishTwitter),SAMPLE_SIZE)]
sampleNews <- englishNews[sample(1:length(englishNews),SAMPLE_SIZE)]
sampleBlogs <- englishBlogs[sample(1:length(englishBlogs),SAMPLE_SIZE)]
textSample <- c(sampleTwitter,sampleNews,sampleBlogs)

writeLines(textSample, "./MilestoneReport/textSample.txt")
theSampleCon <- file("./MilestoneReport/textSample.txt")
theSample <- readLines(theSampleCon)
close(theSampleCon)

# File Sizes:
englishTwitterSize <- round(file.info("final/en_US/en_US.twitter.txt")$size / (1024*1024),0)
englishNewsSize <- round(file.info("final/en_US/en_US.news.txt")$size / (1024*1024),0)
englishBlogsSize <- round(file.info("final/en_US/en_US.blogs.txt")$size / (1024*1024),0)
englishSampleFileSize <- round(file.info("MilestoneReport/textSample.txt")$size / (1024*1024),0)

# Line Counts:
numEnglishTwitterLines <- countLines("final/en_US/en_US.twitter.txt")[1]
numEnglishNewsLines <- countLines("final/en_US/en_US.news.txt")[1]
numEnglishBlogsLines <- countLines("final/en_US/en_US.blogs.txt")[1]
numEnglishSampleLines <- countLines("MilestoneReport/textSample.txt")[1]

# Word Counts:
numWordsEnglishTwitter <- as.numeric(system2("wc", args = "-w < ./final/en_US/en_US.twitter.txt", stdout=TRUE))
numWordsEnglishNews <- as.numeric(system2("wc", args = "-w < ./final/en_US/en_US.news.txt", stdout=TRUE))
numWordsEnglishBlog <- as.numeric(system2("wc", args = "-w < ./final/en_US/en_US.blogs.txt", stdout=TRUE))
numWordsEnglishSample <- as.numeric(system2("wc", args = "-w < MilestoneReport/textSample.txt", stdout=TRUE))

################# ~~~~~~~~~~~~~~~~~ ######## ~~~~~~~~~~~~~~~~~ #################

fileSummary <- data.frame(
  fileName = c("Blogs","News","Twitter", "Aggregated Sample"),
  fileSize = c(round(englishBlogsSize, digits = 2), 
               round(englishNewsSize,digits = 2), 
               round(englishTwitterSize, digits = 2),
               round(englishSampleFileSize, digits = 2)),
  lineCount = c(numEnglishBlogsLines, numEnglishNewsLines, numEnglishTwitterLines, numEnglishSampleLines),
  wordCount = c(numWordsEnglishBlog, numWordsEnglishNews, numWordsEnglishTwitter, numWordsEnglishSample)                  
)
colnames(fileSummary) <- c("Name", "Size", "Num Lines", "Num Words")

fileSummary

# Setup The Text Mining Class:
cname <- file.path(".", "sample")
finalCorpus <- Corpus(DirSource(cname))

# Convert corpus to lowercase:
finalCorpus <- tm_map(finalCorpus, content_transformer(tolower))

# Remove more transforms:
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
finalCorpus <- tm_map(finalCorpus, toSpace, "/|@|\\|")

# Remove punctuation:
finalCorpus <- tm_map(finalCorpus, removePunctuation)

# Remove numbers:
finalCorpus <- tm_map(finalCorpus, removeNumbers)

# Strip whitespace:
finalCorpus <- tm_map(finalCorpus, stripWhitespace)

# Remove english stop words:
finalCorpus <- tm_map(finalCorpus, removeWords, stopwords("english"))

# Initiate stemming:
finalCorpus <- tm_map(finalCorpus, stemDocument)

# Create the N-Grams:
unigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
unigram <- DocumentTermMatrix(finalCorpus, control = list(tokenize = unigramTokenizer))

bigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
bigram <- DocumentTermMatrix(finalCorpus, control = list(tokenize = bigramTokenizer))

trigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
trigram <- DocumentTermMatrix(finalCorpus, control = list(tokenize = trigramTokenizer))

unigramFreq <- sort(colSums(as.matrix(unigram)), decreasing=TRUE)
unigramWordFreq <- data.frame(word=names(unigramFreq), freq=unigramFreq)

#unigramWordFreq <- unigramWordFreq[order(-$freq),] 

paste("Unigrams - Top 5 highest frequencies")

head(unigramWordFreq,5)

bigramFreq <- sort(colSums(as.matrix(bigram)), decreasing=TRUE)
bigramWordFreq <- data.frame(word=names(bigramFreq), freq=bigramFreq)
paste("Bigrams - Top 5 highest frequencies")

trigramFreq <- sort(colSums(as.matrix(trigram)), decreasing=TRUE)
trigramWordFreq <- data.frame(word=names(trigramFreq), freq=trigramFreq)
paste("Trigrams - Top 5 highest frequencies")

head(trigramWordFreq,5)

# Histogram Plots:
unigramWordFreq %>% filter(freq > 1000) %>% ggplot(aes(word,freq)) +
  geom_bar(stat="identity", colour="#37006b", fill="#a257e9") +
  ggtitle("Unigrams With Frequencies Greater Than 1000") +
  xlab("Unigrams") + ylab("Frequency") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  theme(axis.text=element_text(size=14), axis.title=element_text(size=14,face="bold")) +
  theme(plot.title = element_text(lineheight=1.8, face="bold", vjust=3)) +
  theme(plot.margin = unit(c(1,1,1,1), "cm"))

bigramWordFreq %>% filter(freq > 100) %>% ggplot(aes(word,freq)) +
  geom_bar(stat="identity", colour="#990068", fill="#cf6aaf") +
  ggtitle("Bigrams With Frequencies Greater Than 100") +
  xlab("Unigrams") + ylab("Frequency") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  theme(axis.text=element_text(size=14), axis.title=element_text(size=14,face="bold")) +
  theme(plot.title = element_text(lineheight=1.8, face="bold", vjust=3)) +
  #theme(plot.margin = unit(c(1,1,1,1), "cm"))

trigramWordFreq %>% filter(freq > 10) %>% ggplot(aes(word,freq)) +
  geom_bar(stat="identity", colour="#00470d", fill="#4ebc63") +
  ggtitle("Trigrams With Frequencies Greater Than 10") +
  xlab("Trigrams") + ylab("Frequency") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  theme(axis.text=element_text(size=14), axis.title=element_text(size=14,face="bold")) +
  theme(plot.title = element_text(lineheight=1.8, face="bold", vjust=3)) +
  theme(plot.margin = unit(c(1,1,1,1), "cm"))

# Word Clouds:
set.seed(1991)
wordcloud(names(unigramFreq), unigramFreq, max.words=50, scale=c(5, .1), colors=brewer.pal(6, "Set3"))
wordcloud(names(bigramFreq), bigramFreq, max.words=50, scale=c(5, .1), colors=brewer.pal(6, "Set1"))
wordcloud(names(trigramFreq), trigramFreq, max.words=50, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))