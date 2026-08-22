# ECOMMERCE BUSINESS KNOWLEDGE BASE & STORE POLICIES

> **Scope:** Reference this document for all general business logic, shipping rules, payment methods, coupons, and policies. For real-time stock, pricing, and catalog queries, use the dedicated database tools.

---

## 1. Purchase Flow & Requirements

1. **Catalog Search:** Users can browse by name, category, or brand. The catalog only displays items with available stock.
2. **Product Details:** Displays pricing, specifications, and customer reviews.
   - *Requirement:* User authentication (login) is strictly required to add items to the cart.
3. **Cart Management:** Users can modify quantities, remove items, validate discount coupons, and view cost breakdowns (subtotal, applied discounts, shipping cost, and final total).
4. **Shipping Address:** Retrieved automatically from the delivery address configured in the user profile.
5. **Checkout & Confirmation:** The user selects the payment method and places the order. Stock is decremented immediately upon successful checkout confirmation.

---

## 2. Payments & Billing

- **Accepted Payment Methods:**
  - Credit cards (single payment or installments, subject to the customer's card issuer terms).
  - Debit cards.
  - Bank wire / transfer.
  - Cash / Cash on delivery (COD).
- **Processing Time:** Orders are approved and queued for dispatch once the payment provider confirms the transaction.
- **Invoicing & Receipts:** Official fiscal/tax invoices are currently not supported (planned for future releases). The platform order summary and the payment gateway email receipt serve as official proof of purchase.

---

## 3. Shipping & Logistics

- **Currency:** All shipping amounts are in [ARS / USD - Especificar tu moneda local].
- **Flat Shipping Rates:**
  - **Domestic:** $500 flat rate.
  - **International:** $2,500 flat rate (determined automatically by the country set in the user profile).
- **Fulfillment Timelines:**
  - **Dispatch / Handling:** 24 to 48 business hours post-payment accreditation.
  - **Domestic Transit:** 3 to 7 business days after dispatch.
  - **International Transit:** 10 to 20 business days after dispatch.

---

## 4. Discounts & Coupon Rules

- **Redemption:** Coupons must be entered and validated inside the cart prior to final checkout.
- **Available Coupons:**
  - `DESC10` / `PROMO10`: 10% off the product subtotal.
  - `OFF500` / `DESCUENTO`: $500 fixed amount off the product subtotal.
- **Validation Constraints:**
  - Strictly one (1) coupon per order (coupons are non-stackable).
  - Discounts apply exclusively to products; discounts never apply to shipping costs.
  - Subtotal after discounts cannot be less than $0.

---

## 5. Frequently Asked Questions (FAQ)

- **How do I confirm my order went through?**
  Upon checkout completion, an order confirmation screen is displayed, and the order appears in your account purchase history.
- **Can I purchase items that are out of stock?**
  No. The system prevents adding quantities exceeding current verified inventory.
- **How can I update my shipping destination?**
  Update your address inside your user profile settings before completing the checkout process.

---

## 6. Strict Store Policies & Communication Directives

- **Customer Support:** This virtual assistant is the sole and primary support channel. There is no live human support desk.
- **Cancellations, Returns & Refunds:** All sales are final. We do not accept returns, exchanges, order cancellations, or refund requests once an order is placed.
- **Tone Instruction for Policies:** When communicating the absence of refunds, tax invoices, or human agents, maintain a polite, clear, and professional tone without apologizing excessively or making false commitments.

[Language Note: This documentation is in English, but you must reply in the user's preferred language (e.g., Spanish).]