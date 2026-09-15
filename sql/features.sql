-- Feature layer. Kept in SQL so the same logic can run in a warehouse.
-- Input table: applications (raw BAF Base schema). Output: one row per application.
SELECT
    fraud_bool,
    month,
    customer_age,
    income,
    name_email_similarity,
    credit_risk_score,
    proposed_credit_limit,
    intended_balcon_amount,
    days_since_request,
    session_length_in_minutes,
    -- BAF uses -1 as "missing"; make that explicit rather than letting the model learn a magic number
    CASE WHEN prev_address_months_count < 0 THEN NULL ELSE prev_address_months_count END AS prev_address_months,
    CASE WHEN current_address_months_count < 0 THEN NULL ELSE current_address_months_count END AS current_address_months,
    CASE WHEN bank_months_count < 0 THEN NULL ELSE bank_months_count END AS bank_months,
    CASE WHEN device_distinct_emails_8w < 0 THEN NULL ELSE device_distinct_emails_8w END AS device_distinct_emails_8w,
    prev_address_months_count < 0 AS prev_address_missing,
    -- velocity ratios: short-window activity relative to the 4-week baseline
    velocity_6h / NULLIF(velocity_4w, 0) AS velocity_ratio_6h_4w,
    velocity_24h / NULLIF(velocity_4w, 0) AS velocity_ratio_24h_4w,
    velocity_6h,
    velocity_24h,
    velocity_4w,
    zip_count_4w,
    bank_branch_count_8w,
    date_of_birth_distinct_emails_4w,
    device_fraud_count,
    -- identity-verification flags
    email_is_free,
    phone_home_valid,
    phone_mobile_valid,
    (phone_home_valid + phone_mobile_valid) AS phones_valid,
    has_other_cards,
    foreign_request,
    keep_alive_session,
    -- categoricals (encoded downstream)
    payment_type,
    employment_status,
    housing_status,
    source,
    device_os
FROM applications
