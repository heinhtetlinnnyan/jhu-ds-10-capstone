# ngram_engine.R
# Shared functions for the improved next-word model.
# Used by improve_model.R, the report, and later the Shiny app.
#
# Words are stored as integer IDs, and each context (up to 3 previous words)
# as one number. This is smaller and faster than storing words as text.

suppressPackageStartupMessages({
        library(data.table)
        library(stringi)
})
setNumericRounding(0)   # exact joins on numeric context keys

MAX_ORDER <- 4          # 1- to 4-grams; context is up to 3 words

## ---- Text preparation ----------------------------------------------------

clean_text <- function(x) {
        x |>
                stri_trans_tolower() |>
                stri_replace_all_regex("https?://\\S+|www\\.\\S+", " ") |>
                stri_replace_all_regex("[\\u2018\\u2019]", "'") |>
                stri_replace_all_regex("[^a-z' ]", " ") |>
                stri_replace_all_regex("(^| )'+|'+( |$)", " ") |>
                stri_replace_all_regex(" +", " ") |>
                stri_trim_both()
}

# One element per word, with the line it came from
tokenize <- function(lines) {
        toks <- stri_split_fixed(clean_text(lines), " ", omit_empty = TRUE)
        list(w = unlist(toks, use.names = FALSE),
             line = rep.int(seq_along(toks), lengths(toks)))
}

# ID 1 is <unk>; other words ordered by frequency
build_vocab <- function(w, min_count) {
        cnt <- data.table(w = w)[, .N, by = w][N >= min_count]
        setorder(cnt, -N)
        c("<unk>", cnt$w)
}

encode <- function(w, vocab) {
        id <- chmatch(w, vocab)
        id[is.na(id)] <- 1L
        id
}

# Combine word IDs (oldest first) into one numeric key
make_key <- function(parts, base) {
        key <- as.numeric(parts[[1]])
        for (m in seq_along(parts)[-1]) key <- key * base + parts[[m]]
        key
}

key_base <- function(vocab) {
        base <- length(vocab) + 1
        if (base^(MAX_ORDER - 1) >= 2^53)
                stop("Vocabulary too large for exact numeric keys; raise VOCAB_MIN.")
        base
}

## ---- Building the model --------------------------------------------------

# Counts of (context key, next word ID); n-grams never cross line boundaries
count_ngrams <- function(ids, line, n, base) {
        i <- seq_len(length(ids) - n + 1L)
        i <- i[line[i] == line[i + n - 1L]]
        ctx <- if (n == 1) numeric(length(i)) else
                make_key(lapply(0:(n - 2), function(k) ids[i + k]), base)
        data.table(ctx = ctx, word = ids[i + n - 1L])[, .N, by = .(ctx, word)]
}

# pred: top-k next words per context (what the app needs)
# full: all kept n-grams with probabilities (for perplexity only)
build_model <- function(counts, vocab, min_count, top_k = 5) {
        full <- pred <- ctx <- vector("list", MAX_ORDER)
        for (n in seq_len(MAX_ORDER)) {
                dt <- if (n == 1) copy(counts[[1]]) else counts[[n]][N >= min_count]
                dt[, p := N / sum(N), by = ctx]
                full[[n]] <- setkey(dt[, .(ctx, word, p)], ctx, word)
                ctx[[n]]  <- unique(dt$ctx)
                top <- dt[word != 1L, .(ctx, word, p)]
                setorder(top, ctx, -p)
                pred[[n]] <- setkey(top[rowid(ctx) <= top_k], ctx)
        }
        list(pred = pred, full = full, ctx = ctx, vocab = vocab, base = key_base(vocab))
}

app_part <- function(model) model[c("pred", "vocab", "base")]

## ---- Prediction ----------------------------------------------------------

# Context key for order n from columns w1 (last word), w2, w3; NA if too short
ctx_key <- function(cases, n, base) {
        if (n == 1) return(numeric(nrow(cases)))
        parts <- lapply(rev(seq_len(n - 1)), function(k) cases[[paste0("w", k)]])
        make_key(parts, base)
}

# cases: data.table with id, w1, w2, w3 (integer IDs, NA where missing)
# Returns top-k per id with score, raw probability p and n-gram order used.
# Attribute "matched" gives the longest context order found for each id.
predict_batch <- function(model, cases, k = 3, alpha = 0.4) {
        parts <- list()
        for (n in MAX_ORDER:2) {
                q <- data.table(id = cases$id, ctx = ctx_key(cases, n, model$base))[!is.na(ctx)]
                hit <- model$pred[[n]][q, on = "ctx", nomatch = 0L, allow.cartesian = TRUE]
                if (nrow(hit)) parts[[length(parts) + 1]] <-
                        hit[, .(id, word, score = alpha^(MAX_ORDER - n) * p, p, order = n)]
        }
        u <- model$pred[[1]]
        parts[[length(parts) + 1]] <- data.table(
                id    = rep(cases$id, each = nrow(u)),
                word  = rep(u$word, times = nrow(cases)),
                score = rep(alpha^(MAX_ORDER - 1) * u$p, times = nrow(cases)),
                p     = rep(u$p, times = nrow(cases)),
                order = 1L
        )
        cand <- rbindlist(parts)
        matched <- cand[, .(matched = max(order)), by = id]
        setorder(cand, id, -score)
        cand <- unique(cand, by = c("id", "word"))
        cand[, rank := rowid(id)]
        out <- cand[rank <= k]
        setattr(out, "matched", matched)
        out
}

# Phrase in, k suggested words out. Also returns each word's probability
# estimate as an attribute, for showing confidence in the app.
predict_next <- function(model, phrase, k = 3, alpha = 0.4) {
        w <- tokenize(phrase)$w
        ids <- rev(encode(w, model$vocab))
        cases <- data.table(id = 1L,
                            w1 = as.integer(ids[1]),
                            w2 = as.integer(ids[2]),
                            w3 = as.integer(ids[3]))
        res <- predict_batch(model, cases, k, alpha)
        words <- model$vocab[res$word]
        attr(words, "p") <- res$p
        words
}

## ---- Evaluation ----------------------------------------------------------

# Test positions: predict word j (j >= 2 within its line) from up to 3 earlier words
make_cases <- function(ids, line, j, source = NULL) {
        pos  <- rowid(line)
        prev <- function(k) { idx <- j - k; idx[pos[j] <= k] <- NA; ids[idx] }
        dt <- data.table(id = seq_along(j), w1 = prev(1), w2 = prev(2), w3 = prev(3),
                         target = ids[j])
        if (!is.null(source)) dt[, source := source[j]]
        dt
}

# Per case: rank of the true word (NA if not in top k), the first guess and
# its probability (conf), and the longest context matched
score_cases <- function(model, cases, k = 3, alpha = 0.4) {
        pred <- predict_batch(model, cases, k, alpha)
        hit  <- pred[cases[, .(id, target)], on = .(id, word = target), nomatch = 0L]
        out  <- copy(cases)
        out[, rank := NA_integer_]
        out[hit$id, rank := hit$rank]
        out[pred[rank == 1L, .(id, first = word, conf = p)], on = "id",
            `:=`(first = i.first, conf = i.conf)]
        out[attr(pred, "matched"), on = "id", matched := i.matched]
        out
}

top_k_acc <- function(rank, k) mean(!is.na(rank) & rank <= k)

# Interpolated probability of the true word, for perplexity
interp_prob <- function(model, cases, lambda = c(0.1, 0.2, 0.3, 0.4)) {
        num <- den <- numeric(nrow(cases))
        for (n in seq_len(MAX_ORDER)) {
                key <- ctx_key(cases, n, model$base)
                has <- !is.na(key) & key %in% model$ctx[[n]]
                p <- model$full[[n]][.(key, cases$target), on = .(ctx, word), p]
                p[is.na(p)] <- 0
                num <- num + has * lambda[n] * p
                den <- den + has * lambda[n]
        }
        num / den
}

perplexity <- function(model, cases) exp(-mean(log(interp_prob(model, cases))))

size_mb <- function(x) as.numeric(object.size(x)) / 1024^2

ms_per_prediction <- function(model, phrases, alpha = 0.4) {
        secs <- system.time(for (ph in phrases) predict_next(model, ph, alpha = alpha))[["elapsed"]]
        1000 * secs / length(phrases)
}
