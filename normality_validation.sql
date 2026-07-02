SET hive.exec.parallel=true;
SET hive.auto.convert.join=true;

-- Replace your_table with the real table name.
-- Required fields:
--   task_id    : task id
--   pt_d       : date partition, for example 20260606
--   req_cnt    : traffic count
--   time_slice : time slice id, 0-47

SET hivevar:eval_start_pt_d=20260606;
SET hivevar:eval_end_pt_d=20260606;
SET hivevar:hist_days=7;
SET hivevar:min_hist_day_cnt=3;
SET hivevar:min_sample_cnt=100;

-- This script validates the log-space normality assumptions used by the
-- smoothing-correction formula.
--
-- Metrics checked:
--   log_true_correction = LOG(actual_remain / base_remain)
--     This corresponds to u = log(Z), the true log correction factor.
--
--   log_obs_residual = LOG((obs_cum + 1) / (base_cum + 1))
--                      - LOG(actual_remain / base_remain)
--     This corresponds to observation noise in log space.
--
-- Normal reference after standardization:
--   skewness ~= 0
--   excess_kurtosis ~= 0
--   within_1sd_rate ~= 0.6827
--   within_2sd_rate ~= 0.9545
--   within_3sd_rate ~= 0.9973
--   z_p01 ~= -2.326, z_p05 ~= -1.645, z_p25 ~= -0.674
--   z_p50 ~=  0.000, z_p75 ~=  0.674, z_p95 ~=  1.645, z_p99 ~= 2.326

WITH eval_slice AS
(
    SELECT
        task_id,
        CAST(pt_d AS STRING) AS pt_d,
        CAST(time_slice AS INT) AS time_slice,
        SUM(req_cnt) AS req_cnt
    FROM your_table
    WHERE CAST(pt_d AS STRING) BETWEEN '${hivevar:eval_start_pt_d}' AND '${hivevar:eval_end_pt_d}'
      AND CAST(time_slice AS INT) BETWEEN 0 AND 47
    GROUP BY
        task_id,
        CAST(pt_d AS STRING),
        CAST(time_slice AS INT)
),
hist_slice AS
(
    SELECT
        task_id,
        CAST(pt_d AS STRING) AS pt_d,
        CAST(time_slice AS INT) AS time_slice,
        SUM(req_cnt) AS req_cnt
    FROM your_table
    WHERE CAST(pt_d AS STRING) BETWEEN
        REGEXP_REPLACE(
            DATE_SUB(
                FROM_UNIXTIME(
                    UNIX_TIMESTAMP('${hivevar:eval_start_pt_d}', 'yyyyMMdd'),
                    'yyyy-MM-dd'
                ),
                ${hivevar:hist_days}
            ),
            '-',
            ''
        )
        AND '${hivevar:eval_end_pt_d}'
      AND CAST(time_slice AS INT) BETWEEN 0 AND 47
    GROUP BY
        task_id,
        CAST(pt_d AS STRING),
        CAST(time_slice AS INT)
),
base_slice AS
(
    SELECT
        a.task_id,
        a.pt_d,
        a.time_slice,
        a.req_cnt,
        COALESCE(AVG(h.req_cnt), 0.0) AS base_req_cnt,
        COUNT(h.pt_d) AS hist_day_cnt
    FROM eval_slice a
    LEFT JOIN hist_slice h
      ON  a.task_id = h.task_id
      AND a.time_slice = h.time_slice
      AND h.pt_d < a.pt_d
      AND h.pt_d >= REGEXP_REPLACE(
            DATE_SUB(
                FROM_UNIXTIME(
                    UNIX_TIMESTAMP(a.pt_d, 'yyyyMMdd'),
                    'yyyy-MM-dd'
                ),
                ${hivevar:hist_days}
            ),
            '-',
            ''
        )
    GROUP BY
        a.task_id,
        a.pt_d,
        a.time_slice,
        a.req_cnt
),
base_with_hist_cnt AS
(
    SELECT
        task_id,
        pt_d,
        time_slice,
        req_cnt,
        base_req_cnt,
        MIN(hist_day_cnt) OVER (
            PARTITION BY task_id, pt_d
        ) AS min_hist_day_cnt
    FROM base_slice
),
pred AS
(
    SELECT
        task_id,
        pt_d,
        time_slice,
        req_cnt,
        base_req_cnt,
        min_hist_day_cnt,
        SUM(req_cnt) OVER (
            PARTITION BY task_id, pt_d
            ORDER BY time_slice
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS obs_cum,
        SUM(base_req_cnt) OVER (
            PARTITION BY task_id, pt_d
            ORDER BY time_slice
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS base_cum,
        COALESCE(
            SUM(req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            ),
            0.0
        ) AS actual_remain,
        COALESCE(
            SUM(base_req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            ),
            0.0
        ) AS base_remain
    FROM base_with_hist_cnt
),
valid_samples AS
(
    SELECT
        task_id,
        pt_d,
        time_slice,
        obs_cum,
        base_cum,
        actual_remain,
        base_remain,
        LOG(actual_remain / base_remain) AS log_true_correction,
        LOG((obs_cum + 1.0) / (base_cum + 1.0)) AS log_obs_ratio,
        LOG((obs_cum + 1.0) / (base_cum + 1.0))
        -
        LOG(actual_remain / base_remain) AS log_obs_residual
    FROM pred
    WHERE time_slice BETWEEN 0 AND 46
      AND min_hist_day_cnt >= ${hivevar:min_hist_day_cnt}
      AND actual_remain > 0
      AND base_remain > 0
      AND base_cum > 0
),
metric_samples AS
(
    SELECT
        task_id,
        pt_d,
        time_slice,
        base_cum,
        base_remain,
        'log_true_correction' AS metric_name,
        log_true_correction AS metric_value
    FROM valid_samples

    UNION ALL

    SELECT
        task_id,
        pt_d,
        time_slice,
        base_cum,
        base_remain,
        'log_obs_ratio' AS metric_name,
        log_obs_ratio AS metric_value
    FROM valid_samples

    UNION ALL

    SELECT
        task_id,
        pt_d,
        time_slice,
        base_cum,
        base_remain,
        'log_obs_residual' AS metric_name,
        log_obs_residual AS metric_value
    FROM valid_samples
),
segmented_samples AS
(
    SELECT
        metric_name,
        'ALL' AS segment_type,
        'ALL' AS segment_value,
        metric_value
    FROM metric_samples

    UNION ALL

    SELECT
        metric_name,
        'time_slice' AS segment_type,
        CAST(time_slice AS STRING) AS segment_value,
        metric_value
    FROM metric_samples

    UNION ALL

    SELECT
        metric_name,
        'base_cum_bucket' AS segment_type,
        CASE
            WHEN base_cum < 100 THEN '[0,100)'
            WHEN base_cum < 1000 THEN '[100,1000)'
            WHEN base_cum < 10000 THEN '[1000,10000)'
            WHEN base_cum < 100000 THEN '[10000,100000)'
            ELSE '[100000,+)'
        END AS segment_value,
        metric_value
    FROM metric_samples

    UNION ALL

    SELECT
        metric_name,
        'base_remain_bucket' AS segment_type,
        CASE
            WHEN base_remain < 100 THEN '[0,100)'
            WHEN base_remain < 1000 THEN '[100,1000)'
            WHEN base_remain < 10000 THEN '[1000,10000)'
            WHEN base_remain < 100000 THEN '[10000,100000)'
            ELSE '[100000,+)'
        END AS segment_value,
        metric_value
    FROM metric_samples
),
basic_stats AS
(
    SELECT
        metric_name,
        segment_type,
        segment_value,
        COUNT(1) AS sample_cnt,
        AVG(metric_value) AS mean_value,
        STDDEV_SAMP(metric_value) AS std_value
    FROM segmented_samples
    GROUP BY
        metric_name,
        segment_type,
        segment_value
    HAVING COUNT(1) >= ${hivevar:min_sample_cnt}
),
normality_stats AS
(
    SELECT
        b.metric_name,
        b.segment_type,
        b.segment_value,
        b.sample_cnt,
        b.mean_value,
        b.std_value,

        AVG(POWER((s.metric_value - b.mean_value) / b.std_value, 3)) AS skewness,
        AVG(POWER((s.metric_value - b.mean_value) / b.std_value, 4)) - 3.0 AS excess_kurtosis,

        AVG(
            CASE
                WHEN ABS((s.metric_value - b.mean_value) / b.std_value) <= 1.0 THEN 1.0
                ELSE 0.0
            END
        ) AS within_1sd_rate,
        AVG(
            CASE
                WHEN ABS((s.metric_value - b.mean_value) / b.std_value) <= 2.0 THEN 1.0
                ELSE 0.0
            END
        ) AS within_2sd_rate,
        AVG(
            CASE
                WHEN ABS((s.metric_value - b.mean_value) / b.std_value) <= 3.0 THEN 1.0
                ELSE 0.0
            END
        ) AS within_3sd_rate,

        (PERCENTILE_APPROX(s.metric_value, 0.01) - b.mean_value) / b.std_value AS z_p01,
        (PERCENTILE_APPROX(s.metric_value, 0.05) - b.mean_value) / b.std_value AS z_p05,
        (PERCENTILE_APPROX(s.metric_value, 0.25) - b.mean_value) / b.std_value AS z_p25,
        (PERCENTILE_APPROX(s.metric_value, 0.50) - b.mean_value) / b.std_value AS z_p50,
        (PERCENTILE_APPROX(s.metric_value, 0.75) - b.mean_value) / b.std_value AS z_p75,
        (PERCENTILE_APPROX(s.metric_value, 0.95) - b.mean_value) / b.std_value AS z_p95,
        (PERCENTILE_APPROX(s.metric_value, 0.99) - b.mean_value) / b.std_value AS z_p99
    FROM segmented_samples s
    JOIN basic_stats b
      ON  s.metric_name = b.metric_name
      AND s.segment_type = b.segment_type
      AND s.segment_value = b.segment_value
    WHERE b.std_value > 0
    GROUP BY
        b.metric_name,
        b.segment_type,
        b.segment_value,
        b.sample_cnt,
        b.mean_value,
        b.std_value
)
SELECT
    metric_name,
    segment_type,
    segment_value,
    sample_cnt,
    mean_value,
    std_value,
    skewness,
    excess_kurtosis,
    within_1sd_rate,
    within_2sd_rate,
    within_3sd_rate,
    z_p01,
    z_p05,
    z_p25,
    z_p50,
    z_p75,
    z_p95,
    z_p99,
    CASE
        WHEN ABS(skewness) <= 0.5
         AND ABS(excess_kurtosis) <= 1.0
         AND ABS(within_1sd_rate - 0.6827) <= 0.08
         AND ABS(within_2sd_rate - 0.9545) <= 0.05
        THEN 'roughly_normal'
        ELSE 'need_segment_or_non_normal'
    END AS normality_hint
FROM normality_stats
ORDER BY
    metric_name,
    segment_type,
    segment_value
;
