# improve_model.R
# Stage 2: try to improve accuracy while keeping the model small and fast.
#
# Changes tested against a baseline, all on the same held-out test set:
#   1. Integer word IDs instead of text keys (size and speed).
#   2. Equal lines from blogs, news and Twitter (the pooled sample is mostly Twitter).
#   3. More training data, with stronger pruning to hold size down.
#   4. Backoff weight (alpha) tuned on a separate development set.
# Then: which words the model misses and why, and whether its confidence
# matches how often it is right.
#
# Run from the folder that contains data/ and ngram_engine.R.

source("ngram_engine.R")
set.seed(2026)

## ---- Settings ------------------------------------------------------------

DATA_DIR <- "data/en_US"
FILES <- c(blogs   = "en_US.blogs.txt",
           news    = "en_US.news.txt",
           twitter = "en_US.twitter.txt")

TEST_LINES_PER_SOURCE <- 2000   # never used for training or tuning
DEV_LINES_PER_SOURCE  <- 1000   # used only to tune alpha
TEST_CASES_PER_SOURCE <- 7000
DEV_CASES_PER_SOURCE  <- 3000
N_TIMING   <- 300
VOCAB_MIN  <- 2
TOP_K      <- 5
ALPHAS     <- seq(0.1, 0.9, by = 0.1)

VARIANTS <- list(
        list(name = "Baseline: pooled sample, 3%",        sampling = "pooled",   frac = 0.03, min_count = 2),
        list(name = "Balanced sources, 3%",               sampling = "balanced", frac = 0.03, min_count = 2),
        list(name = "Balanced sources, 10%, min count 3", sampling = "balanced", frac = 0.10, min_count = 3)
)
# alpha is tuned on the last variant, which is reported again with the tuned value

dir.create("models", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)

## ---- Load and split by source --------------------------------------------

read_corpus <- function(path) {
        con <- file(path, open = "rb")
        on.exit(close(con))
        readLines(con, encoding = "UTF-8", skipNul = TRUE)
}

paths <- file.path(DATA_DIR, FILES)
if (!all(file.exists(paths)))
        stop("Data files not found under ", normalizePath(DATA_DIR, mustWork = FALSE),
             ". Run this script from the folder that contains data/.")

splits <- lapply(paths, function(p) {
        x   <- read_corpus(p)
        idx <- sample(length(x))
        n_t <- TEST_LINES_PER_SOURCE
        n_d <- DEV_LINES_PER_SOURCE
        list(test = x[idx[seq_len(n_t)]],
             dev  = x[idx[n_t + seq_len(n_d)]],
             pool = x[idx[-seq_len(n_t + n_d)]])      # already shuffled
})
names(splits) <- names(FILES)
total_lines <- sum(sapply(splits, function(s) length(s$test) + length(s$dev) + length(s$pool)))
invisible(gc())

# Training lines for a variant
get_train_lines <- function(sampling, frac) {
        n_total <- round(frac * total_lines)
        if (sampling == "balanced") {
                n_each <- round(n_total / length(splits))
                unlist(lapply(splits, function(s) s$pool[seq_len(min(n_each, length(s$pool)))]),
                       use.names = FALSE)
        } else {
                pool_all <- unlist(lapply(splits, `[[`, "pool"), use.names = FALSE)
                pool_all[sample.int(length(pool_all), n_total)]
        }
}

# Held-out text, tokenized, with the source of every word
prep_heldout <- function(part) {
        txt <- unlist(lapply(splits, `[[`, part), use.names = FALSE)
        src <- rep(names(splits), times = sapply(splits, function(s) length(s[[part]])))
        tok <- tokenize(txt)
        tok$src <- src[tok$line]
        tok
}

# Same number of test positions from each source
pick_positions <- function(tok, per_source) {
        eligible <- which(rowid(tok$line) >= 2)
        by_src <- split(eligible, tok$src[eligible])
        sort(unlist(lapply(by_src, function(e)
                if (length(e) > per_source) e[sample.int(length(e), per_source)] else e),
                use.names = FALSE))
}

test <- prep_heldout("test")
dev  <- prep_heldout("dev")
test_pos <- pick_positions(test, TEST_CASES_PER_SOURCE)
dev_pos  <- pick_positions(dev,  DEV_CASES_PER_SOURCE)

timing <- make_cases(test$w, test$line, sample(test_pos, N_TIMING))
phrases <- with(timing, stri_trim_both(paste(fifelse(is.na(w3), "", w3),
                                             fifelse(is.na(w2), "", w2), w1)))

## ---- Build and evaluate each variant -------------------------------------

summarise_scores <- function(sc) {
        sc[, .(top1 = top_k_acc(rank, 1), top2 = top_k_acc(rank, 2), top3 = top_k_acc(rank, 3))]
}

results   <- list()
by_source <- list()

record <- function(name, v, model, sc, pp, ms, build_sec, alpha) {
        results[[length(results) + 1]] <<- data.table(
                variant     = name,
                sampling    = v$sampling,
                train_frac  = v$frac,
                min_count   = v$min_count,
                alpha       = alpha,
                vocab_size  = length(model$vocab),
                app_mb      = round(size_mb(app_part(model)), 1),
                build_sec   = round(build_sec, 1),
                ms_per_pred = round(ms, 2),
                summarise_scores(sc),
                perplexity  = round(pp, 1)
        )
        by_source[[length(by_source) + 1]] <<-
                sc[, c(.(variant = name), summarise_scores(.SD)), by = source]
}

for (v in VARIANTS) {
        message("Building: ", v$name)
        t0 <- Sys.time()
        tok    <- tokenize(get_train_lines(v$sampling, v$frac))
        vocab  <- build_vocab(tok$w, VOCAB_MIN)
        ids    <- encode(tok$w, vocab)
        base   <- key_base(vocab)
        counts <- lapply(seq_len(MAX_ORDER), function(n) count_ngrams(ids, tok$line, n, base))
        model  <- build_model(counts, vocab, v$min_count, TOP_K)
        build_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        rm(tok, ids, counts); invisible(gc())
        
        test_cases <- make_cases(encode(test$w, vocab), test$line, test_pos, test$src)
        sc <- score_cases(model, test_cases)
        pp <- perplexity(model, test_cases)
        ms <- ms_per_prediction(model, phrases)
        record(v$name, v, model, sc, pp, ms, build_sec, 0.4)
        message(sprintf("  top3 %.3f | pp %.1f | %.1f MB | %.2f ms",
                        top_k_acc(sc$rank, 3), pp, size_mb(app_part(model)), ms))
}
# `model`, `vocab`, `test_cases`, `pp`, `build_sec` now belong to the last variant

## ---- Tune alpha on the development set -----------------------------------

dev_cases <- make_cases(encode(dev$w, vocab), dev$line, dev_pos, dev$src)
alpha_tab <- rbindlist(lapply(ALPHAS, function(a) {
        s <- score_cases(model, dev_cases, alpha = a)
        data.table(alpha = a, summarise_scores(s))
}))
best_alpha <- alpha_tab[which.max(top3), alpha]
message("Best alpha on dev set: ", best_alpha)

v_last <- VARIANTS[[length(VARIANTS)]]
sc_final <- score_cases(model, test_cases, alpha = best_alpha)
ms_final <- ms_per_prediction(model, phrases, alpha = best_alpha)
record(paste0(v_last$name, ", tuned alpha"), v_last, model, sc_final, pp, ms_final,
       build_sec, best_alpha)

## ---- What does the final model miss, and why? ----------------------------

word_of <- function(id) ifelse(is.na(id), NA_character_, model$vocab[id])

miss <- sc_final[is.na(rank)]
missed_reasons <- miss[, .(reason = fcase(
        target == 1L,  "Next word is rare (outside vocabulary)",
        matched == 1L, "Earlier words never seen together in training",
        default = "Phrase seen, but the word was not in the top 3"))][
                , .(cases = .N), by = reason][
                        , `:=`(pct_of_misses = round(100 * cases / nrow(miss), 1),
                               pct_of_all    = round(100 * cases / nrow(sc_final), 1))][order(-cases)]

missed_words <- sc_final[target != 1L,
                         .(occurrences = .N, missed = sum(is.na(rank))), by = target][
                                 order(-missed)][1:15][
                                         , `:=`(word = word_of(target), miss_rate = round(missed / occurrences, 2))][
                                                 , .(word, occurrences, missed, miss_rate)]

missed_pairs <- miss[target != 1L & !is.na(w1),
                     .(missed = .N,
                       usual_guess = first[which.max(tabulate(match(first, first)))]),
                     by = .(w1, target)][order(-missed)][1:15][
                             , .(pair = paste(word_of(w1), word_of(target)), missed, usual_guess = word_of(usual_guess))]

## ---- How reliable is the model's confidence? -----------------------------

sc_final[, conf_bin := cut(conf, breaks = c(0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1),
                           include.lowest = TRUE)]
calibration <- sc_final[, .(cases     = .N,
                            mean_conf = round(mean(conf), 3),
                            hit_rate  = round(top_k_acc(rank, 1), 3)),
                        by = conf_bin][order(conf_bin)]

## ---- Save ----------------------------------------------------------------

results   <- rbindlist(results)
by_source <- rbindlist(by_source)
print(results)

fwrite(results,        "output/improved_results.csv")
fwrite(by_source,      "output/improved_by_source.csv")
fwrite(alpha_tab,      "output/alpha_tuning.csv")
fwrite(missed_reasons, "output/missed_reasons.csv")
fwrite(missed_words,   "output/missed_words.csv")
fwrite(missed_pairs,   "output/missed_pairs.csv")
fwrite(calibration,    "output/calibration.csv")

final <- app_part(model)
final$alpha <- best_alpha
final$top3  <- top_k_acc(sc_final$rank, 3)   # shown in the app
saveRDS(final, "models/final_model.rds")
message("Saved models/final_model.rds")                                              
