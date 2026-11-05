install.packages("R.utils")
library(R.utils)
file_path_twitter <- "data/en_US/en_US.twitter.txt"

length(readLines(file_path_twitter))

file_paths <- c(
        "data/en_US/en_US.blogs.txt",
        "data/en_US/en_US.news.txt",
        "data/en_US/en_US.twitter.txt")

max_length <- max(sapply(file_paths, function(path) {
        lines <- readLines(path, warn = FALSE)
        max(nchar(lines))
}))

max_length

longest_per_file <- sapply(file_paths, function(path) {
        lines <- readLines(path, warn = FALSE)
        max(nchar(lines))
})
longest_per_file


twitterlines <- readLines(file_path_twitter)
sum(grepl("love", twitterlines, fixed = TRUE)) / sum(grepl("hate", twitterlines, fixed = TRUE))

twitter <- readLines(file_path_twitter, warn = FALSE)
grep("biostats", twitter, value = TRUE)
sum(twitter == "A computer once beat me at chess, but it was no match for me at kickboxing")


library(tidyverse)
set.seed(1234)
twitter_raw <- readLines("data/en_US/en_US.twitter.txt", warn = FALSE)
blogs_raw <- readLines("data/en_US/en_US.blogs.txt", warn = FALSE)
news_raw <- readLines("data/en_US/en_US.news.txt", warn = FALSE)

twitter_sample <- sample(twitter_raw, length(twitter_raw) * 0.01)
blogs_sample <- sample(blogs_raw, length(blogs_raw) * 0.01)
news_sample <- sample(news_raw, length(news_raw) * 0.01)

sample_df <- tibble(
        source = c(rep("twitter", length(twitter_sample)),
                   rep("blogs",   length(blogs_sample)),
                   rep("news",    length(news_sample))),
        text = c(twitter_sample, blogs_sample, news_sample)
)

## Tokenize

library(tidytext)

# Break text into single words (unigrams)
words_clean <- sample_df %>%
        # Remove special characters / punctuation but keep spaces
        mutate(text = gsub("[^a-zA-Z' ]", "", text)) %>%
        # Tokenize into lowercased words
        unnest_tokens(word, text)

# See top 10 most frequent words overall
words_clean %>%
        count(word, sort = TRUE) %>%
        head(10)

bigrams <- sample_df %>%
        unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
        drop_na() %>%
        count(bigram, sort = TRUE)

head(bigrams, 10)


trigrams <- sample_df %>%
        unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
        drop_na() %>%
        count(trigram, sort = TRUE)

head(trigrams, 10)


# Plot top 15 bigrams across the entire sample
bigrams %>%
        slice_max(n, n = 15) %>%
        ggplot(aes(x = reorder(bigram, n), y = n)) +
        geom_col(fill = "steelblue") +
        coord_flip() +
        labs(
                title = "Top 15 Most Common Bigrams (Word Pairs)",
                x = "Bigram",
                y = "Frequency Count"
        ) +
        theme_minimal()



library(tidyverse)
library(tidytext)

# 1. Get word frequencies sorted from most to least common
word_counts <- words_clean %>%
        count(word, sort = TRUE) %>%
        mutate(
                cumulative_words = cumsum(n),
                total_words = sum(n),
                coverage = cumulative_words / total_words
        )

# 2. How many unique words cover 50% of the dataset?
words_50 <- word_counts %>%
        filter(coverage >= 0.50) %>%
        slice(1) %>%
        pull(cumulative_words) # or simply row number: which(word_counts$coverage >= 0.50)[1]

count_50 <- which(word_counts$coverage >= 0.50)[1]
count_90 <- which(word_counts$coverage >= 0.90)[1]

cat("Unique words for 50% coverage:", count_50, "\n")
cat("Unique words for 90% coverage:", count_90, "\n")


install.packages("hunspell")

# Install and use hunspell
library(hunspell)

# Check which words are not in standard English dictionary
non_english <- words_clean %>%
        distinct(word) %>%
        mutate(is_english = hunspell_check(word)) %>%
        filter(!is_english)

object.size(twitter_raw)
gc()

install.packages("stringi")




