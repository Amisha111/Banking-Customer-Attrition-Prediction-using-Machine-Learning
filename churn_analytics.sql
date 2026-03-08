-- ============================================================
--  BANKING CUSTOMER CHURN PREDICTION
--  SQL Analytics Queries (PostgreSQL / MySQL compatible)
-- ============================================================
--  Author  : Portfolio Project
--  Dataset : bank_churn (10,000 retail banking customers)
--  Purpose : Production-ready SQL for churn analytics
-- ============================================================


-- ────────────────────────────────────────────────────────────
--  SECTION 1 — DATA QUALITY & OVERVIEW
-- ────────────────────────────────────────────────────────────

-- 1.1 Dataset Overview
SELECT
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned_customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct,
    ROUND(AVG(Age), 1)                                AS avg_age,
    ROUND(AVG(CreditScore), 0)                        AS avg_credit_score,
    ROUND(AVG(Balance), 2)                            AS avg_balance,
    ROUND(AVG(EstimatedSalary), 2)                    AS avg_salary,
    ROUND(AVG(Tenure), 1)                             AS avg_tenure_years
FROM bank_churn;


-- 1.2 Null Check
SELECT
    SUM(CASE WHEN CustomerID       IS NULL THEN 1 ELSE 0 END) AS null_customerid,
    SUM(CASE WHEN CreditScore      IS NULL THEN 1 ELSE 0 END) AS null_creditscore,
    SUM(CASE WHEN Age              IS NULL THEN 1 ELSE 0 END) AS null_age,
    SUM(CASE WHEN Balance          IS NULL THEN 1 ELSE 0 END) AS null_balance,
    SUM(CASE WHEN Exited           IS NULL THEN 1 ELSE 0 END) AS null_exited
FROM bank_churn;


-- ────────────────────────────────────────────────────────────
--  SECTION 2 — CHURN RATE BY SEGMENT
-- ────────────────────────────────────────────────────────────

-- 2.1 Churn Rate by Geography
SELECT
    Geography,
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct,
    ROUND(AVG(Balance), 2)                            AS avg_balance,
    ROUND(AVG(EstimatedSalary), 2)                    AS avg_salary
FROM bank_churn
GROUP BY Geography
ORDER BY churn_rate_pct DESC;


-- 2.2 Churn Rate by Age Group
SELECT
    CASE
        WHEN Age BETWEEN 18 AND 25 THEN '18–25'
        WHEN Age BETWEEN 26 AND 35 THEN '26–35'
        WHEN Age BETWEEN 36 AND 45 THEN '36–45'
        WHEN Age BETWEEN 46 AND 55 THEN '46–55'
        WHEN Age BETWEEN 56 AND 65 THEN '56–65'
        ELSE '65+'
    END                                               AS age_group,
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct
FROM bank_churn
GROUP BY age_group
ORDER BY age_group;


-- 2.3 Churn Rate by Number of Products
SELECT
    NumOfProducts,
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct,
    ROUND(AVG(Balance), 2)                            AS avg_balance
FROM bank_churn
GROUP BY NumOfProducts
ORDER BY NumOfProducts;


-- 2.4 Churn Rate by Activity Status & Gender
SELECT
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS membership_status,
    Gender,
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct
FROM bank_churn
GROUP BY membership_status, Gender
ORDER BY churn_rate_pct DESC;


-- 2.5 Churn Rate by Tenure Cohort
SELECT
    CASE
        WHEN Tenure BETWEEN 0 AND 2  THEN 'New (0–2 yrs)'
        WHEN Tenure BETWEEN 3 AND 5  THEN 'Growing (3–5 yrs)'
        WHEN Tenure BETWEEN 6 AND 8  THEN 'Established (6–8 yrs)'
        ELSE 'Loyal (9–10 yrs)'
    END                                               AS tenure_cohort,
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct,
    ROUND(AVG(MonthlyTransactions), 1)                AS avg_monthly_txns
FROM bank_churn
GROUP BY tenure_cohort
ORDER BY churn_rate_pct DESC;


-- 2.6 Churn Rate by Credit Score Tier
SELECT
    CASE
        WHEN CreditScore < 500                  THEN '< 500  (Very Poor)'
        WHEN CreditScore BETWEEN 500 AND 579    THEN '500-579 (Poor)'
        WHEN CreditScore BETWEEN 580 AND 669    THEN '580-669 (Fair)'
        WHEN CreditScore BETWEEN 670 AND 739    THEN '670-739 (Good)'
        WHEN CreditScore BETWEEN 740 AND 799    THEN '740-799 (Very Good)'
        ELSE '800+   (Exceptional)'
    END                                               AS credit_tier,
    COUNT(*)                                          AS total_customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct
FROM bank_churn
GROUP BY credit_tier
ORDER BY churn_rate_pct DESC;


-- ────────────────────────────────────────────────────────────
--  SECTION 3 — HIGH-RISK CUSTOMER IDENTIFICATION
-- ────────────────────────────────────────────────────────────

-- 3.1 High-Risk Customer Scoring (Rule-Based)
SELECT
    CustomerID,
    Geography,
    Gender,
    Age,
    Balance,
    NumOfProducts,
    IsActiveMember,
    ComplaintFiled,
    SatisfactionScore,
    Exited,
    -- Risk score: sum of risk flags
    (
        CASE WHEN Age > 55                              THEN 2 ELSE 0 END +
        CASE WHEN IsActiveMember = 0                    THEN 3 ELSE 0 END +
        CASE WHEN NumOfProducts >= 3                    THEN 2 ELSE 0 END +
        CASE WHEN Geography = 'Germany'                 THEN 1 ELSE 0 END +
        CASE WHEN CreditScore < 500                     THEN 2 ELSE 0 END +
        CASE WHEN Balance = 0                           THEN 1 ELSE 0 END +
        CASE WHEN Tenure < 2                            THEN 1 ELSE 0 END +
        CASE WHEN ComplaintFiled = 1                    THEN 4 ELSE 0 END +
        CASE WHEN SatisfactionScore <= 2                THEN 2 ELSE 0 END +
        CASE WHEN MonthlyTransactions < 2              THEN 2 ELSE 0 END
    )                                                  AS risk_score,
    CASE
        WHEN (
            CASE WHEN Age > 55 THEN 2 ELSE 0 END +
            CASE WHEN IsActiveMember = 0 THEN 3 ELSE 0 END +
            CASE WHEN NumOfProducts >= 3 THEN 2 ELSE 0 END +
            CASE WHEN Geography = 'Germany' THEN 1 ELSE 0 END +
            CASE WHEN CreditScore < 500 THEN 2 ELSE 0 END +
            CASE WHEN Balance = 0 THEN 1 ELSE 0 END +
            CASE WHEN Tenure < 2 THEN 1 ELSE 0 END +
            CASE WHEN ComplaintFiled = 1 THEN 4 ELSE 0 END +
            CASE WHEN SatisfactionScore <= 2 THEN 2 ELSE 0 END +
            CASE WHEN MonthlyTransactions < 2 THEN 2 ELSE 0 END
        ) >= 8 THEN 'CRITICAL'
        WHEN (
            CASE WHEN Age > 55 THEN 2 ELSE 0 END +
            CASE WHEN IsActiveMember = 0 THEN 3 ELSE 0 END +
            CASE WHEN NumOfProducts >= 3 THEN 2 ELSE 0 END +
            CASE WHEN Geography = 'Germany' THEN 1 ELSE 0 END +
            CASE WHEN CreditScore < 500 THEN 2 ELSE 0 END +
            CASE WHEN Balance = 0 THEN 1 ELSE 0 END +
            CASE WHEN Tenure < 2 THEN 1 ELSE 0 END +
            CASE WHEN ComplaintFiled = 1 THEN 4 ELSE 0 END +
            CASE WHEN SatisfactionScore <= 2 THEN 2 ELSE 0 END +
            CASE WHEN MonthlyTransactions < 2 THEN 2 ELSE 0 END
        ) >= 5 THEN 'HIGH'
        WHEN (
            CASE WHEN Age > 55 THEN 2 ELSE 0 END +
            CASE WHEN IsActiveMember = 0 THEN 3 ELSE 0 END +
            CASE WHEN NumOfProducts >= 3 THEN 2 ELSE 0 END +
            CASE WHEN Geography = 'Germany' THEN 1 ELSE 0 END +
            CASE WHEN CreditScore < 500 THEN 2 ELSE 0 END +
            CASE WHEN Balance = 0 THEN 1 ELSE 0 END +
            CASE WHEN Tenure < 2 THEN 1 ELSE 0 END +
            CASE WHEN ComplaintFiled = 1 THEN 4 ELSE 0 END +
            CASE WHEN SatisfactionScore <= 2 THEN 2 ELSE 0 END +
            CASE WHEN MonthlyTransactions < 2 THEN 2 ELSE 0 END
        ) >= 3 THEN 'MEDIUM'
        ELSE 'LOW'
    END                                                AS risk_tier
FROM bank_churn
ORDER BY risk_score DESC
LIMIT 100;


-- 3.2 Customers with Model-Predicted High Churn Risk
-- (Use after running Python model and loading predictions)
SELECT
    CustomerID,
    Geography,
    Age,
    Balance,
    NumOfProducts,
    IsActiveMember,
    ROUND(RF_ChurnProb * 100, 1)                      AS churn_probability_pct,
    RiskSegment,
    Exited                                            AS actual_churn
FROM churn_predictions
WHERE RF_ChurnProb > 0.70
  AND Exited = 0          -- not yet churned — target for retention
ORDER BY RF_ChurnProb DESC;


-- ────────────────────────────────────────────────────────────
--  SECTION 4 — REVENUE IMPACT ANALYSIS
-- ────────────────────────────────────────────────────────────

-- 4.1 Revenue at Risk by Segment
SELECT
    Geography,
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS status,
    COUNT(CASE WHEN Exited = 1 THEN 1 END)            AS churned_customers,
    ROUND(SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END), 2)
                                                      AS balance_lost,
    ROUND(AVG(CASE WHEN Exited = 1 THEN Balance END), 2)
                                                      AS avg_churned_balance,
    -- Estimated Annual Revenue Lost (assume 1.5% NIM on balance)
    ROUND(SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END) * 0.015, 2)
                                                      AS estimated_annual_revenue_lost
FROM bank_churn
GROUP BY Geography, status
ORDER BY balance_lost DESC;


-- 4.2 Customer Lifetime Value Proxy
SELECT
    CustomerID,
    Geography,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    EstimatedSalary,
    -- Simple CLV proxy: balance × products × tenure_factor
    ROUND(Balance * NumOfProducts * (1 + Tenure * 0.1), 2)
                                                      AS clv_proxy,
    Exited
FROM bank_churn
ORDER BY clv_proxy DESC
LIMIT 50;


-- 4.3 High-Value Churned Customers (Priority Winback List)
SELECT
    CustomerID,
    Geography,
    Age,
    Balance,
    NumOfProducts,
    Tenure,
    SatisfactionScore,
    ComplaintFiled,
    ROUND(Balance * NumOfProducts * (1 + Tenure * 0.1), 2) AS clv_proxy
FROM bank_churn
WHERE Exited = 1
  AND Balance > 100000
ORDER BY clv_proxy DESC
LIMIT 25;


-- ────────────────────────────────────────────────────────────
--  SECTION 5 — COHORT & TREND ANALYSIS
-- ────────────────────────────────────────────────────────────

-- 5.1 Churn Rate by Geography × Gender × Product Matrix
SELECT
    Geography,
    Gender,
    NumOfProducts,
    COUNT(*)                                          AS customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 1)          AS churn_rate_pct,
    ROUND(AVG(Balance), 0)                            AS avg_balance
FROM bank_churn
GROUP BY Geography, Gender, NumOfProducts
HAVING COUNT(*) > 20
ORDER BY churn_rate_pct DESC;


-- 5.2 Active vs Inactive Engagement Deep Dive
SELECT
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS status,
    ROUND(AVG(MonthlyTransactions), 2)                AS avg_monthly_txns,
    ROUND(AVG(LoginFreqMonthly), 2)                   AS avg_logins_monthly,
    ROUND(AVG(Balance), 2)                            AS avg_balance,
    ROUND(AVG(NumOfProducts), 2)                      AS avg_products,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)          AS churn_rate_pct,
    COUNT(*)                                          AS total_customers
FROM bank_churn
GROUP BY status;


-- 5.3 Complaint Impact Analysis
SELECT
    ComplaintFiled,
    SatisfactionScore,
    COUNT(*)                                          AS customers,
    SUM(Exited)                                       AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 1)          AS churn_rate_pct
FROM bank_churn
GROUP BY ComplaintFiled, SatisfactionScore
ORDER BY ComplaintFiled DESC, SatisfactionScore;


-- ────────────────────────────────────────────────────────────
--  SECTION 6 — RETENTION STRATEGY TARGETING
-- ────────────────────────────────────────────────────────────

-- 6.1 Priority Retention Targets
-- Active customers likely to churn — highest intervention ROI
SELECT
    CustomerID,
    Geography,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    MonthlyTransactions,
    SatisfactionScore,
    -- Retention priority score
    ROUND(
        Balance * 0.000005
        + (10 - SatisfactionScore) * 5
        + (5 - LEAST(Tenure, 5)) * 3
        + CASE WHEN NumOfProducts = 1 THEN 5 ELSE 0 END
        + CASE WHEN MonthlyTransactions < 3 THEN 8 ELSE 0 END
    , 1)                                              AS retention_priority_score,
    CASE
        WHEN Balance > 100000 THEN 'Premium — Personal Outreach'
        WHEN Balance > 50000  THEN 'Standard — Email Campaign'
        WHEN Balance > 10000  THEN 'Basic — Loyalty Offer'
        ELSE 'Low Value — Automated'
    END                                               AS recommended_action
FROM bank_churn
WHERE Exited = 0
  AND IsActiveMember = 1
  AND (SatisfactionScore <= 2 OR ComplaintFiled = 1 OR MonthlyTransactions < 3)
ORDER BY retention_priority_score DESC
LIMIT 200;


-- 6.2 Retention Campaign Sizing
SELECT
    CASE
        WHEN Balance > 100000 THEN 'Tier 1 — Premium'
        WHEN Balance > 50000  THEN 'Tier 2 — Standard'
        WHEN Balance > 10000  THEN 'Tier 3 — Basic'
        ELSE 'Tier 4 — Low Value'
    END                                               AS customer_tier,
    COUNT(*)                                          AS total_at_risk,
    SUM(Exited)                                       AS already_churned,
    ROUND(AVG(Balance), 0)                            AS avg_balance,
    ROUND(SUM(Balance), 0)                            AS total_balance_at_risk
FROM bank_churn
WHERE SatisfactionScore <= 3
   OR ComplaintFiled = 1
   OR IsActiveMember = 0
GROUP BY customer_tier
ORDER BY total_balance_at_risk DESC;
