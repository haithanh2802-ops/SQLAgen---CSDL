CREATE TABLE IF NOT EXISTS product_category_translation (
    product_category_name VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS customers (
    customer_id CHAR(32) PRIMARY KEY,
    customer_unique_id CHAR(32) NOT NULL,
    customer_zip_code_prefix CHAR(5) NOT NULL,
    customer_city VARCHAR(100) NOT NULL,
    customer_state CHAR(2) NOT NULL,
    INDEX idx_customers_unique (customer_unique_id),
    INDEX idx_customers_zip (customer_zip_code_prefix),
    INDEX idx_customers_state (customer_state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sellers (
    seller_id CHAR(32) PRIMARY KEY,
    seller_zip_code_prefix CHAR(5) NOT NULL,
    seller_city VARCHAR(100) NOT NULL,
    seller_state CHAR(2) NOT NULL,
    INDEX idx_sellers_zip (seller_zip_code_prefix),
    INDEX idx_sellers_state (seller_state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS products (
    product_id CHAR(32) PRIMARY KEY,
    product_category_name VARCHAR(100),
    product_name_length SMALLINT UNSIGNED,
    product_description_length SMALLINT UNSIGNED,
    product_photos_qty SMALLINT UNSIGNED,
    product_weight_g INT UNSIGNED,
    product_length_cm SMALLINT UNSIGNED,
    product_height_cm SMALLINT UNSIGNED,
    product_width_cm SMALLINT UNSIGNED,
    INDEX idx_products_category (product_category_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS orders (
    order_id CHAR(32) PRIMARY KEY,
    customer_id CHAR(32) NOT NULL,
    order_status VARCHAR(20) NOT NULL,
    order_purchase_timestamp DATETIME NOT NULL,
    order_approved_at DATETIME,
    order_delivered_carrier_date DATETIME,
    order_delivered_customer_date DATETIME,
    order_estimated_delivery_date DATETIME,
    CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    INDEX idx_orders_customer (customer_id),
    INDEX idx_orders_purchase (order_purchase_timestamp),
    INDEX idx_orders_status (order_status),
    INDEX idx_orders_delivery (order_delivered_customer_date, order_estimated_delivery_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS order_items (
    order_id CHAR(32) NOT NULL,
    order_item_id SMALLINT UNSIGNED NOT NULL,
    product_id CHAR(32) NOT NULL,
    seller_id CHAR(32) NOT NULL,
    shipping_limit_date DATETIME NOT NULL,
    price DECIMAL(12,2) NOT NULL,
    freight_value DECIMAL(12,2) NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    CONSTRAINT fk_items_order FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CONSTRAINT fk_items_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT fk_items_seller FOREIGN KEY (seller_id) REFERENCES sellers(seller_id),
    INDEX idx_items_product (product_id),
    INDEX idx_items_seller (seller_id),
    INDEX idx_items_shipping_limit (shipping_limit_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS order_payments (
    order_id CHAR(32) NOT NULL,
    payment_sequential SMALLINT UNSIGNED NOT NULL,
    payment_type VARCHAR(30) NOT NULL,
    payment_installments SMALLINT UNSIGNED NOT NULL,
    payment_value DECIMAL(12,2) NOT NULL,
    PRIMARY KEY (order_id, payment_sequential),
    CONSTRAINT fk_payments_order FOREIGN KEY (order_id) REFERENCES orders(order_id),
    INDEX idx_payments_type (payment_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS order_reviews (
    review_id CHAR(32) NOT NULL,
    order_id CHAR(32) NOT NULL,
    review_score TINYINT UNSIGNED NOT NULL,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date DATETIME NOT NULL,
    review_answer_timestamp DATETIME NOT NULL,
    PRIMARY KEY (review_id, order_id),
    CONSTRAINT fk_reviews_order FOREIGN KEY (order_id) REFERENCES orders(order_id),
    INDEX idx_reviews_order (order_id),
    INDEX idx_reviews_score (review_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS geolocation (
    geolocation_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    geolocation_zip_code_prefix CHAR(5) NOT NULL,
    geolocation_lat DECIMAL(10,7) NOT NULL,
    geolocation_lng DECIMAL(10,7) NOT NULL,
    geolocation_city VARCHAR(100) NOT NULL,
    geolocation_state CHAR(2) NOT NULL,
    INDEX idx_geolocation_zip (geolocation_zip_code_prefix),
    INDEX idx_geolocation_state (geolocation_state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE OR REPLACE VIEW analytics_order_facts AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) AS delivery_days,
    DATEDIFF(o.order_delivered_customer_date, o.order_estimated_delivery_date) AS delay_days,
    CASE
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1
        WHEN o.order_delivered_customer_date IS NOT NULL THEN 0
        ELSE NULL
    END AS is_late,
    COALESCE(i.item_count, 0) AS item_count,
    COALESCE(i.item_revenue, 0) AS item_revenue,
    COALESCE(i.freight_value, 0) AS freight_value,
    COALESCE(p.payment_value, 0) AS payment_value,
    p.max_installments,
    r.review_score
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
LEFT JOIN (
    SELECT order_id, COUNT(*) AS item_count, SUM(price) AS item_revenue,
           SUM(freight_value) AS freight_value
    FROM order_items
    GROUP BY order_id
) AS i ON i.order_id = o.order_id
LEFT JOIN (
    SELECT order_id, SUM(payment_value) AS payment_value,
           MAX(payment_installments) AS max_installments
    FROM order_payments
    GROUP BY order_id
) AS p ON p.order_id = o.order_id
LEFT JOIN (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
) AS r ON r.order_id = o.order_id;

CREATE OR REPLACE VIEW analytics_order_items AS
SELECT
    oi.order_id,
    oi.order_item_id,
    o.order_purchase_timestamp,
    o.order_status,
    oi.product_id,
    COALESCE(t.product_category_name_english, p.product_category_name) AS product_category,
    oi.seller_id,
    s.seller_city,
    s.seller_state,
    oi.price,
    oi.freight_value,
    oi.shipping_limit_date
FROM order_items AS oi
JOIN orders AS o ON o.order_id = oi.order_id
JOIN products AS p ON p.product_id = oi.product_id
JOIN sellers AS s ON s.seller_id = oi.seller_id
LEFT JOIN product_category_translation AS t
  ON t.product_category_name = p.product_category_name;
