SET hive.exec.parallel=true;
SET hive.auto.convert.join=true;

SET hivevar:eval_start_pt_d=20260606;
SET hivevar:eval_end_pt_d=20260606;
SET hivevar:alpha=1000.0;

SELECT
    COUNT(1) AS sample_cnt,

    SUM(actual_remain) AS actual_remain_sum,

    SUM(base_pred_remain) AS base_pred_remain_sum,
    SUM(mvp_pred_remain) AS mvp_pred_remain_sum,

    SUM(ABS(base_pred_remain - actual_remain)) AS base_abs_error,
    SUM(ABS(mvp_pred_remain - actual_remain)) AS mvp_abs_error,

    SUM(ABS(base_pred_remain - actual_remain)) / SUM(actual_remain) AS base_wape,
    SUM(ABS(mvp_pred_remain - actual_remain)) / SUM(actual_remain) AS mvp_wape,

    (
        SUM(ABS(base_pred_remain - actual_remain))
        -
        SUM(ABS(mvp_pred_remain - actual_remain))
    ) / SUM(ABS(base_pred_remain - actual_remain)) AS relative_error_reduction
FROM
(
    SELECT
        task_id,
        pt_d,
        time_slice AS pred_time_slice,

        actual_remain,
        base_remain AS base_pred_remain,

        base_remain
        *
        POW(
            (obs_cum + 1.0) / (base_cum + 1.0),
            base_cum / (base_cum + ${hivevar:alpha})
        ) AS mvp_pred_remain
    FROM
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
        FROM
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
            FROM
            (
                SELECT
                    a.task_id,
                    a.pt_d,
                    a.time_slice,
                    a.req_cnt,

                    COALESCE(AVG(h.req_cnt), 0.0) AS base_req_cnt,
                    COUNT(h.pt_d) AS hist_day_cnt
                FROM
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
                ) a
                LEFT JOIN
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
                                7
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
                ) h
                ON  a.task_id = h.task_id
                AND a.time_slice = h.time_slice
                AND h.pt_d < a.pt_d
                AND h.pt_d >= REGEXP_REPLACE(
                    DATE_SUB(
                        FROM_UNIXTIME(
                            UNIX_TIMESTAMP(a.pt_d, 'yyyyMMdd'),
                            'yyyy-MM-dd'
                        ),
                        7
                    ),
                    '-',
                    ''
                )
                GROUP BY
                    a.task_id,
                    a.pt_d,
                    a.time_slice,
                    a.req_cnt
            ) base_slice
        ) base_with_hist_cnt
    ) pred
    WHERE time_slice BETWEEN 0 AND 46
      AND min_hist_day_cnt >= 3
      AND actual_remain > 0
      AND base_remain > 0
      AND base_cum > 0
) final_pred
;