# next_word_model.R
# Build and evaluate an n-gram next-word model.
#   - Ranking (what the app shows): stupid backoff over 1- to 4-grams.
#   - Perplexity: interpolated maximum-likelihood model on the same tables,
#     because stupid-backoff scores are not probabilities.
# Loops over training size and pruning level, then reports accuracy, perplexity,
# speed and size for each combination.

suppressPackageStartupMessages({
        library(data.table)
        library(stringi)
        library(ggplot2)
})
set.seed(1234)

## ---- Settings ------------------------------------------------------------

DATA_DIR     <- "data/en_US"
MAX_ORDER    <- 4                        # 1- to 4-grams (code assumes 4)
TOP_K        <- 5                        # candidates kept per context
ALPHA        <- 0.4                      # stupid-backoff penalty per step down
LAMBDA       <- c(0.1, 0.2, 0.3, 0.4)    # interpolation weights, orders 1..4
VOCAB_MIN    <- 2                        # words seen fewer times -> <unk>
TEST_LINES   <- 5000                     # held-out lines, never used in training
N_CASES      <- 20000                    # (context, next word) pairs to score
N_TIMING     <- 300                      # single predictions to time
TRAIN_FRACS  <- c(0.01, 0.03, 0.05)      # share of all lines used to train
MIN_COUNTS   <- c(1, 2, 4)               # drop 2- to 4-grams seen fewer times

dir.create("models", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)

## ---- Load and split ------------------------------------------------------

read_corpus <- function(path) {
        con <- file(path, open = "rb")
        on.exit(close(con))
        readLines(con, encoding = "UTF-8", skipNul = TRUE)
}

files <- file.path(DATA_DIR, c("en_US.blogs.txt", "en_US.news.txt", "en_US.twitter.txt"))
if (!all(file.exists(files)))
        stop("Data files not found under ", normalizePath(DATA_DIR, mustWork = FALSE),
             ". Run this script from the folder that contains data/.")
all_lines   <- unlist(lapply(files, read_corpus), use.names = FALSE)
total_lines <- length(all_lines)

shuffled   <- sample(total_lines)
test_raw   <- all_lines[shuffled[seq_len(TEST_LINES)]]
train_pool <- all_lines[shuffled[TEST_LINES + seq_len(round(max(TRAIN_FRACS) * total_lines))]]
rm(all_lines); invisible(gc())

## ---- Cleaning and tokenizing ---------------------------------------------

clean_text <- function(x) {
        x <- stri_trans_tolower(x)
        x <- stri_replace_all_regex(x, "https?://\\S+|www\\.\\S+", " ")
        x <- stri_replace_all_regex(x, "[\\u2018\\u2019]", "'")
        x <- stri_replace_all_regex(x, "[^a-z' ]", " ")
        x <- stri_replace_all_regex(x, "(^| )'+|'+( |$)", " ")   # stray apostrophes
        stri_trim_both(stri_replace_all_regex(x, " +", " "))
}

# Flat representation: one row per word, with the line it came from
tokenize <- function(lines) {
        toks <- stri_split_fixed(clean_text(lines), " ", omit_empty = TRUE)
        list(w = unlist(toks, use.names = FALSE),
             line = rep.int(seq_along(toks), lengths(toks)))
}

map_vocab <- function(w, vocab) {
        w[!(w %chin% vocab)] <- "<unk>"
        w
}

train <- tokenize(train_pool)
test  <- tokenize(test_raw)
rm(train_pool, test_raw); invisible(gc())

## ---- N-gram counting -----------------------------------------------------

# Returns prefix (first n-1 words), word (last word), N (count).
# N-grams never cross line boundaries.
build_ngrams <- function(w, line, n) {
        i <- seq_len(length(w) - n + 1)
        i <- i[line[i] == line[i + n - 1]]
        prefix <- if (n == 1) rep("", length(i)) else
                do.call(paste, lapply(0:(n - 2), function(k) w[i + k]))
        data.table(prefix = prefix, word = w[i + n - 1])[, .N, by = .(prefix, word)]
}

# full: probabilities for perplexity; pred: top-K per context for ranking
build_model <- function(counts, vocab, min_count) {
        full <- pred <- ctx <- vector("list", MAX_ORDER)
        for (n in seq_len(MAX_ORDER)) {
                dt <- if (n == 1) copy(counts[[1]]) else counts[[n]][N >= min_count]
                dt[, p := N / sum(N), by = prefix]     # normalised after pruning
                full[[n]] <- setkey(dt[, .(prefix, word, p)], prefix, word)
                ctx[[n]]  <- unique(dt$prefix)
                top <- dt[word != "<unk>", .(prefix, word, p)]
                setorder(top, prefix, -p)
                pred[[n]] <- setkey(top[rowid(prefix) <= TOP_K], prefix)
        }
        list(full = full, pred = pred, ctx = ctx, vocab = vocab)
}

## ---- Prediction ----------------------------------------------------------

# Context key for order n, oldest word first; NA if the line is too short
ctx_key <- function(cases, n) {
        if (n == 1) return(rep("", nrow(cases)))
        parts <- lapply(rev(seq_len(n - 1)), function(k) cases[[paste0("w", k)]])
        key <- do.call(paste, parts)
        key[Reduce(`|`, lapply(parts, is.na))] <- NA_character_
        key
}

# cases: data.table with id, w1 (last word), w2, w3. Returns top k per id.
predict_batch <- function(model, cases, k = 3) {
        parts <- list()
        for (n in MAX_ORDER:2) {
                q <- data.table(id = cases$id, prefix = ctx_key(cases, n))[!is.na(prefix)]
                hit <- model$pred[[n]][q, on = "prefix", nomatch = 0L, allow.cartesian = TRUE]
                if (nrow(hit)) parts[[length(parts) + 1]] <-
                        hit[, .(id, word, score = ALPHA^(MAX_ORDER - n) * p)]
        }
        u <- model$pred[[1]]
        parts[[length(parts) + 1]] <- data.table(
                id    = rep(cases$id, each = nrow(u)),
                word  = rep(u$word, times = nrow(cases)),
                score = rep(ALPHA^(MAX_ORDER - 1) * u$p, times = nrow(cases))
        )
        cand <- rbindlist(parts)[, .(score = max(score)), by = .(id, word)]
        setorder(cand, id, -score)
        cand[, rank := rowid(id)][rank <= k]
}

# Single phrase -> character vector of k suggestions (what the app will call)
predict_next <- function(model, phrase, k = 3) {
        w <- stri_split_fixed(clean_text(phrase), " ", omit_empty = TRUE)[[1]]
        w <- rev(map_vocab(w, model$vocab))
        cases <- data.table(id = 1L,
                            w1 = as.character(w[1]),
                            w2 = as.character(w[2]),
                            w3 = as.character(w[3]))
        predict_batch(model, cases, k)$word
}

## ---- Evaluation ----------------------------------------------------------

# Each test case: predict word j from up to 3 previous words (j >= 2 in line)
cases_from <- function(w, line, j) {
        pos  <- rowid(line)
        prev <- function(k) { idx <- j - k; idx[pos[j] <= k] <- NA; w[idx] }
        data.table(id = seq_along(j), w1 = prev(1), w2 = prev(2), w3 = prev(3), target = w[j])
}

# Share of cases where the true word is in the top 1, top 2, top 3
accuracy_at <- function(model, cases, k = 3) {
        pred <- predict_batch(model, cases, k)
        hit  <- pred[cases, on = .(id, word = target), nomatch = 0L]
        r <- rep(NA_integer_, nrow(cases))
        r[hit$id] <- hit$rank
        sapply(seq_len(k), function(i) mean(!is.na(r) & r <= i))
}

# Interpolated probability of the target; orders whose context is unseen
# are skipped and the remaining weights rescaled to sum to 1
interp_prob <- function(model, cases) {
        num <- den <- numeric(nrow(cases))
        for (n in seq_len(MAX_ORDER)) {
                key <- ctx_key(cases, n)
                has <- !is.na(key) & key %chin% model$ctx[[n]]
                p <- model$full[[n]][.(key, cases$target), on = .(prefix, word), p]
                p[is.na(p)] <- 0
                num <- num + has * LAMBDA[n] * p
                den <- den + has * LAMBDA[n]
        }
        num / den
}

time_predictions <- function(model, phrases) {
        secs <- system.time(for (ph in phrases) predict_next(model, ph))[["elapsed"]]
        1000 * secs / length(phrases)
}

size_mb <- function(x) as.numeric(object.size(x)) / 1024^2

# Fixed test positions, shared by every model
pos_test <- rowid(test$line)
eligible <- which(pos_test >= 2)
case_idx <- sort(sample(eligible, min(N_CASES, length(eligible))))

timing_cases <- cases_from(test$w, test$line, sample(case_idx, N_TIMING))
phrases <- with(timing_cases,
                paste(fifelse(is.na(w3), "", w3), fifelse(is.na(w2), "", w2), w1))

## ---- Main loop -----------------------------------------------------------

results <- list()

for (frac in TRAIN_FRACS) {
        n_lines <- round(frac * total_lines)
        keep    <- train$line <= n_lines
        w_tr    <- train$w[keep]
        l_tr    <- train$line[keep]
        
        t0 <- Sys.time()
        vocab  <- data.table(w = w_tr)[, .N, by = w][N >= VOCAB_MIN, w]
        w_tr   <- map_vocab(w_tr, vocab)
        counts <- lapply(seq_len(MAX_ORDER), function(n) build_ngrams(w_tr, l_tr, n))
        count_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        
        cases <- cases_from(map_vocab(test$w, vocab), test$line, case_idx)
        
        for (mc in MIN_COUNTS) {
                t1    <- Sys.time()
                model <- build_model(counts, vocab, mc)
                build_sec <- count_sec + as.numeric(difftime(Sys.time(), t1, units = "secs"))
                
                acc <- accuracy_at(model, cases)
                pp  <- exp(-mean(log(interp_prob(model, cases))))
                ms  <- time_predictions(model, phrases)
                
                app_model <- model[c("pred", "vocab")]
                saveRDS(app_model, sprintf("models/model_f%.2f_mc%d.rds", frac, mc))
                
                results[[length(results) + 1]] <- data.table(
                        train_frac  = frac,
                        train_lines = n_lines,
                        min_count   = mc,
                        vocab_size  = length(vocab),
                        ngrams      = sum(sapply(model$full, nrow)),
                        app_mb      = round(size_mb(app_model), 1),
                        full_mb     = round(size_mb(model$full), 1),
                        build_sec   = round(build_sec, 1),
                        ms_per_pred = round(ms, 2),
                        top1        = round(acc[1], 3),
                        top2        = round(acc[2], 3),
                        top3        = round(acc[3], 3),
                        perplexity  = round(pp, 1)
                )
                message(sprintf("frac %.2f | min_count %d | top3 %.3f | pp %.1f | %.1f MB | %.2f ms",
                                frac, mc, acc[3], pp, size_mb(app_model), ms))
                rm(model, app_model); invisible(gc())
        }
        rm(counts); invisible(gc())
}

results <- rbindlist(results)
print(results)
fwrite(results, "output/model_results.csv")

## ---- Summary plots -------------------------------------------------------

results[, config := sprintf("%d%% / min %d", round(100 * train_frac), min_count)]

p_size <- ggplot(results, aes(app_mb, top3, colour = factor(train_frac))) +
        geom_point(size = 3) +
        geom_text(aes(label = paste0("min ", min_count)), vjust = -1, size = 3, show.legend = FALSE) +
        labs(x = "Model size (MB)", y = "Top-3 accuracy", colour = "Training share",
             title = "Accuracy vs model size") +
        theme_minimal()
ggsave("output/accuracy_vs_size.png", p_size, width = 7, height = 4.5)

p_speed <- ggplot(results, aes(ms_per_pred, top3, colour = factor(train_frac))) +
        geom_point(size = 3) +
        labs(x = "Milliseconds per prediction", y = "Top-3 accuracy", colour = "Training share",
             title = "Accuracy vs speed") +
        theme_minimal()
ggsave("output/accuracy_vs_speed.png", p_speed, width = 7, height = 4.5)

p_pp <- ggplot(results, aes(perplexity, top3, colour = factor(train_frac))) +
        geom_point(size = 3) +
        labs(x = "Perplexity (lower is better)", y = "Top-3 accuracy", colour = "Training share",
             title = "Does perplexity track accuracy?") +
        theme_minimal()
ggsave("output/perplexity_vs_accuracy.png", p_pp, width = 7, height = 4.5)

cat("\nCorrelation, perplexity vs top-1:", round(cor(results$perplexity, results$top1), 2))
cat("\nCorrelation, perplexity vs top-3:", round(cor(results$perplexity, results$top3), 2), "\n")