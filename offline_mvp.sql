SELECT
    COUNT(1) AS sample_cnt,

    SUM(actual_remain) AS actual_remain_sum,

    SUM(base_pred_remain) AS base_pred_remain_sum,
    SUM(win12_pred_remain) AS win12_pred_remain_sum,
    SUM(win24_pred_remain) AS win24_pred_remain_sum,
    SUM(win48_pred_remain) AS win48_pred_remain_sum,

    SUM(ABS(base_pred_remain - actual_remain)) / SUM(actual_remain) AS base_wape,
    SUM(ABS(win12_pred_remain - actual_remain)) / SUM(actual_remain) AS win12_wape,
    SUM(ABS(win24_pred_remain - actual_remain)) / SUM(actual_remain) AS win24_wape,
    SUM(ABS(win48_pred_remain - actual_remain)) / SUM(actual_remain) AS win48_wape,

    (
        SUM(ABS(base_pred_remain - actual_remain))
        -
        SUM(ABS(win12_pred_remain - actual_remain))
    ) / SUM(ABS(base_pred_remain - actual_remain)) AS win12_error_reduction,

    (
        SUM(ABS(base_pred_remain - actual_remain))
        -
        SUM(ABS(win24_pred_remain - actual_remain))
    ) / SUM(ABS(base_pred_remain - actual_remain)) AS win24_error_reduction,

    (
        SUM(ABS(base_pred_remain - actual_remain))
        -
        SUM(ABS(win48_pred_remain - actual_remain))
    ) / SUM(ABS(base_pred_remain - actual_remain)) AS win48_error_reduction

FROM
(
    SELECT
        task_id,
        pt_d,
        time_slice,

        actual_remain,
        base_remain AS base_pred_remain,

        base_remain
        *
        POW(
            (obs_win12 + 1.0) / (base_win12 + 1.0),
            base_win12 / (base_win12 + 1000.0)
        ) AS win12_pred_remain,

        base_remain
        *
        POW(
            (obs_win24 + 1.0) / (base_win24 + 1.0),
            base_win24 / (base_win24 + 1000.0)
        ) AS win24_pred_remain,

        base_remain
        *
        POW(
            (obs_win48 + 1.0) / (base_win48 + 1.0),
            base_win48 / (base_win48 + 1000.0)
        ) AS win48_pred_remain

    FROM
    (
        SELECT
            task_id,
            pt_d,
            time_slice,
            req_cnt,
            base_req_cnt,

            SUM(req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
            ) AS obs_win12,

            SUM(base_req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
            ) AS base_win12,

            SUM(req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS obs_win24,

            SUM(base_req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS base_win24,

            SUM(req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 47 PRECEDING AND CURRENT ROW
            ) AS obs_win48,

            SUM(base_req_cnt) OVER (
                PARTITION BY task_id, pt_d
                ORDER BY time_slice
                ROWS BETWEEN 47 PRECEDING AND CURRENT ROW
            ) AS base_win48,

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
                a.task_id,
                a.pt_d,
                a.time_slice,
                a.req_cnt,
                COALESCE(AVG(h.req_cnt), 0.0) AS base_req_cnt
            FROM
            (
                SELECT
                    task_id,
                    pt_d,
                    CAST(time_slice AS INT) AS time_slice,
                    SUM(req_cnt) AS req_cnt
                FROM your_table
                WHERE pt_d = '20260606'
                  AND CAST(time_slice AS INT) BETWEEN 0 AND 47
                GROUP BY
                    task_id,
                    pt_d,
                    CAST(time_slice AS INT)
            ) a
            LEFT JOIN
            (
                SELECT
                    task_id,
                    pt_d,
                    CAST(time_slice AS INT) AS time_slice,
                    SUM(req_cnt) AS req_cnt
                FROM your_table
                WHERE pt_d BETWEEN '20260530' AND '20260605'
                  AND CAST(time_slice AS INT) BETWEEN 0 AND 47
                GROUP BY
                    task_id,
                    pt_d,
                    CAST(time_slice AS INT)
            ) h
            ON  a.task_id = h.task_id
            AND a.time_slice = h.time_slice
            GROUP BY
                a.task_id,
                a.pt_d,
                a.time_slice,
                a.req_cnt
        ) base_data
    ) win_data
    WHERE time_slice BETWEEN 0 AND 46
      AND actual_remain > 0
      AND base_remain > 0
) final_data
;