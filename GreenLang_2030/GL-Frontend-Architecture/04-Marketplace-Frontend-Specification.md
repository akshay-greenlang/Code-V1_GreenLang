# GreenLang Marketplace Frontend Specification
## Commercial Platform for Premium Agents & Services

### 1. Technical Architecture

#### Core Stack
```yaml
Frontend:
  Framework: Next.js 14 with Commerce
  Language: TypeScript 5.3+
  UI: Tailwind CSS + Headless UI
  State: Zustand + React Query
  Payments: Stripe Elements + PayPal SDK
  Auth: NextAuth.js + Auth0

Backend Integration:
  API: GraphQL (Apollo Client)
  Subscriptions: GraphQL Subscriptions
  Real-time: WebSockets
  File Storage: AWS S3
  CDN: CloudFront

Commerce Features:
  Cart: Persistent cart with Redis
  Checkout: Multi-step with validation
  Payments: Stripe, PayPal, Wire Transfer
  Invoicing: Automatic generation
  Tax: Integrated tax calculation
  Licensing: Digital license management
```

### 2. Marketplace Home Page

```typescript
export const MarketplaceHome: React.FC = () => {
  const { data: featured } = useQuery({
    queryKey: ['featured-products'],
    queryFn: getFeaturedProducts,
  });

  const { data: categories } = useQuery({
    queryKey: ['marketplace-categories'],
    queryFn: getMarketplaceCategories,
  });

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 text-white">
        <div className="container mx-auto px-4 py-20">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl font-bold mb-6">
              GreenLang Marketplace
            </h1>
            <p className="text-xl mb-8 opacity-90">
              Premium climate intelligence agents, enterprise solutions, and professional services
            </p>

            {/* Search Bar */}
            <div className="max-w-2xl mx-auto">
              <MarketplaceSearch />
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-4 gap-8 mt-12">
              <div>
                <div className="text-3xl font-bold">2,500+</div>
                <div className="text-sm opacity-75">Premium Agents</div>
              </div>
              <div>
                <div className="text-3xl font-bold">500+</div>
                <div className="text-sm opacity-75">Enterprise Solutions</div>
              </div>
              <div>
                <div className="text-3xl font-bold">98%</div>
                <div className="text-sm opacity-75">Customer Satisfaction</div>
              </div>
              <div>
                <div className="text-3xl font-bold">24/7</div>
                <div className="text-sm opacity-75">Support Available</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Products Carousel */}
      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-3xl font-bold">Featured Solutions</h2>
            <Link
              href="/marketplace/featured"
              className="text-green-600 hover:text-green-700 font-medium"
            >
              View all →
            </Link>
          </div>

          <ProductCarousel products={featured?.products || []} />
        </div>
      </section>

      {/* Category Grid */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-8">Browse by Category</h2>
          <CategoryGrid categories={categories || []} />
        </div>
      </section>

      {/* Enterprise Solutions */}
      <section className="py-16 bg-gradient-to-br from-gray-900 to-gray-800 text-white">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-6">Enterprise Solutions</h2>
            <p className="text-lg mb-8 opacity-90">
              Custom agents, white-label platforms, and dedicated support for your organization
            </p>
            <div className="flex gap-4 justify-center">
              <Link
                href="/marketplace/enterprise"
                className="px-6 py-3 bg-white text-gray-900 rounded-lg hover:bg-gray-100"
              >
                Explore Solutions
              </Link>
              <Link
                href="/contact-sales"
                className="px-6 py-3 border-2 border-white rounded-lg hover:bg-white hover:text-gray-900"
              >
                Contact Sales
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Indicators */}
      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Trusted by Industry Leaders</h2>
            <p className="text-lg text-gray-600">
              Fortune 500 companies rely on GreenLang for climate intelligence
            </p>
          </div>
          <ClientLogos />
        </div>
      </section>
    </div>
  );
};
```

### 3. Product Listing Page

```typescript
interface Product {
  id: string;
  name: string;
  description: string;
  category: string;
  vendor: Vendor;
  pricing: {
    model: 'one-time' | 'subscription' | 'usage-based';
    price: number;
    currency: string;
    billingPeriod?: 'monthly' | 'yearly';
    tiers?: PricingTier[];
  };
  features: string[];
  requirements: string[];
  rating: number;
  reviews: number;
  sales: number;
  images: string[];
  demo?: string;
  trial?: boolean;
}

export const ProductCard: React.FC<{ product: Product }> = ({ product }) => {
  const [isInCart, setIsInCart] = useState(false);
  const [showQuickView, setShowQuickView] = useState(false);
  const { addToCart } = useCart();

  const handleAddToCart = () => {
    addToCart(product);
    setIsInCart(true);
    toast.success(`${product.name} added to cart`);
  };

  return (
    <>
      <div className="bg-white rounded-lg border hover:shadow-xl transition-all group">
        {/* Product Image */}
        <div className="relative aspect-video overflow-hidden rounded-t-lg bg-gray-100">
          <img
            src={product.images[0]}
            alt={product.name}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
          />

          {/* Badges */}
          <div className="absolute top-4 left-4 flex gap-2">
            {product.trial && (
              <Badge className="bg-blue-600 text-white">Free Trial</Badge>
            )}
            {product.sales > 1000 && (
              <Badge className="bg-green-600 text-white">Bestseller</Badge>
            )}
          </div>

          {/* Quick Actions */}
          <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="flex flex-col gap-2">
              <button
                onClick={() => setShowQuickView(true)}
                className="p-2 bg-white rounded-lg shadow hover:shadow-md"
              >
                <Eye className="w-4 h-4" />
              </button>
              {product.demo && (
                <Link
                  href={product.demo}
                  className="p-2 bg-white rounded-lg shadow hover:shadow-md"
                >
                  <Play className="w-4 h-4" />
                </Link>
              )}
            </div>
          </div>
        </div>

        {/* Product Info */}
        <div className="p-6">
          {/* Vendor */}
          <div className="flex items-center gap-2 mb-3">
            <img
              src={product.vendor.logo}
              alt={product.vendor.name}
              className="w-6 h-6 rounded"
            />
            <span className="text-sm text-gray-600">{product.vendor.name}</span>
            {product.vendor.verified && (
              <CheckCircle className="w-4 h-4 text-blue-500" />
            )}
          </div>

          {/* Name & Description */}
          <h3 className="font-semibold text-lg mb-2">
            <Link
              href={`/marketplace/products/${product.id}`}
              className="hover:text-green-600"
            >
              {product.name}
            </Link>
          </h3>
          <p className="text-gray-600 text-sm mb-4 line-clamp-2">
            {product.description}
          </p>

          {/* Features */}
          <div className="flex flex-wrap gap-2 mb-4">
            {product.features.slice(0, 3).map((feature) => (
              <span
                key={feature}
                className="text-xs px-2 py-1 bg-gray-100 rounded"
              >
                {feature}
              </span>
            ))}
            {product.features.length > 3 && (
              <span className="text-xs px-2 py-1 text-gray-500">
                +{product.features.length - 3} more
              </span>
            )}
          </div>

          {/* Rating */}
          <div className="flex items-center gap-2 mb-4">
            <RatingStars rating={product.rating} />
            <span className="text-sm text-gray-600">
              {product.rating.toFixed(1)} ({product.reviews})
            </span>
            <span className="text-sm text-gray-400">•</span>
            <span className="text-sm text-gray-600">
              {product.sales} sales
            </span>
          </div>

          {/* Pricing */}
          <div className="flex items-end justify-between mb-4">
            <div>
              <div className="text-2xl font-bold">
                {product.pricing.currency}
                {product.pricing.price}
              </div>
              {product.pricing.model === 'subscription' && (
                <div className="text-sm text-gray-600">
                  per {product.pricing.billingPeriod}
                </div>
              )}
            </div>
            {product.pricing.tiers && (
              <Link
                href={`/marketplace/products/${product.id}#pricing`}
                className="text-sm text-green-600 hover:underline"
              >
                View pricing tiers
              </Link>
            )}
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={handleAddToCart}
              disabled={isInCart}
              className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
            >
              {isInCart ? 'In Cart' : 'Add to Cart'}
            </button>
            {product.trial && (
              <Link
                href={`/marketplace/products/${product.id}/trial`}
                className="px-4 py-2 border border-green-600 text-green-600 rounded-lg hover:bg-green-50"
              >
                Try Free
              </Link>
            )}
          </div>
        </div>
      </div>

      {/* Quick View Modal */}
      {showQuickView && (
        <ProductQuickView
          product={product}
          onClose={() => setShowQuickView(false)}
        />
      )}
    </>
  );
};
```

### 4. Shopping Cart & Checkout

```typescript
// Shopping Cart Component
export const ShoppingCart: React.FC = () => {
  const { items, total, removeItem, updateQuantity, clearCart } = useCart();
  const [promoCode, setPromoCode] = useState('');
  const [discount, setDiscount] = useState<Discount | null>(null);

  const applyPromoCode = async () => {
    try {
      const result = await validatePromoCode(promoCode);
      setDiscount(result);
      toast.success('Promo code applied!');
    } catch (error) {
      toast.error('Invalid promo code');
    }
  };

  const subtotal = items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const discountAmount = discount ? calculateDiscount(subtotal, discount) : 0;
  const tax = calculateTax(subtotal - discountAmount);
  const finalTotal = subtotal - discountAmount + tax;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Shopping Cart</h1>

      {items.length === 0 ? (
        <EmptyCart />
      ) : (
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Cart Items */}
          <div className="lg:col-span-2 space-y-4">
            {items.map((item) => (
              <CartItem
                key={item.id}
                item={item}
                onUpdateQuantity={(quantity) =>
                  updateQuantity(item.id, quantity)
                }
                onRemove={() => removeItem(item.id)}
              />
            ))}
          </div>

          {/* Order Summary */}
          <div className="bg-gray-50 rounded-lg p-6 h-fit">
            <h2 className="text-xl font-semibold mb-4">Order Summary</h2>

            {/* Promo Code */}
            <div className="mb-6">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={promoCode}
                  onChange={(e) => setPromoCode(e.target.value)}
                  placeholder="Promo code"
                  className="flex-1 px-3 py-2 border rounded"
                />
                <button
                  onClick={applyPromoCode}
                  className="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-900"
                >
                  Apply
                </button>
              </div>
            </div>

            {/* Price Breakdown */}
            <div className="space-y-3 mb-6">
              <div className="flex justify-between">
                <span>Subtotal</span>
                <span>${subtotal.toFixed(2)}</span>
              </div>

              {discount && (
                <div className="flex justify-between text-green-600">
                  <span>Discount ({discount.code})</span>
                  <span>-${discountAmount.toFixed(2)}</span>
                </div>
              )}

              <div className="flex justify-between">
                <span>Tax</span>
                <span>${tax.toFixed(2)}</span>
              </div>

              <div className="pt-3 border-t font-semibold text-lg">
                <div className="flex justify-between">
                  <span>Total</span>
                  <span>${finalTotal.toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Checkout Button */}
            <Link
              href="/checkout"
              className="block w-full px-6 py-3 bg-green-600 text-white text-center rounded-lg hover:bg-green-700"
            >
              Proceed to Checkout
            </Link>

            {/* Security Badge */}
            <div className="mt-4 flex items-center justify-center gap-2 text-sm text-gray-600">
              <Lock className="w-4 h-4" />
              <span>Secure checkout with SSL encryption</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Checkout Flow
export const CheckoutFlow: React.FC = () => {
  const [step, setStep] = useState<'billing' | 'payment' | 'review'>('billing');
  const [billingInfo, setBillingInfo] = useState<BillingInfo>({});
  const [paymentMethod, setPaymentMethod] = useState<'card' | 'paypal' | 'wire'>('card');
  const { items, total } = useCart();

  const steps = [
    { id: 'billing', label: 'Billing Information', icon: 'user' },
    { id: 'payment', label: 'Payment Method', icon: 'credit-card' },
    { id: 'review', label: 'Review & Confirm', icon: 'check' },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-center">
          {steps.map((s, index) => (
            <React.Fragment key={s.id}>
              <div
                className={`
                  flex items-center gap-3 px-4 py-2 rounded-lg
                  ${step === s.id ? 'bg-green-100 text-green-700' : ''}
                  ${steps.indexOf(steps.find(st => st.id === step)!) > index
                    ? 'text-green-600'
                    : 'text-gray-400'}
                `}
              >
                <div
                  className={`
                    w-8 h-8 rounded-full flex items-center justify-center
                    ${step === s.id
                      ? 'bg-green-600 text-white'
                      : steps.indexOf(steps.find(st => st.id === step)!) > index
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-300'}
                  `}
                >
                  {steps.indexOf(steps.find(st => st.id === step)!) > index ? (
                    <Check className="w-4 h-4" />
                  ) : (
                    index + 1
                  )}
                </div>
                <span className="font-medium">{s.label}</span>
              </div>
              {index < steps.length - 1 && (
                <div className="w-12 h-0.5 bg-gray-300" />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="max-w-4xl mx-auto">
        {/* Billing Information Step */}
        {step === 'billing' && (
          <BillingForm
            initialData={billingInfo}
            onSubmit={(data) => {
              setBillingInfo(data);
              setStep('payment');
            }}
          />
        )}

        {/* Payment Method Step */}
        {step === 'payment' && (
          <PaymentForm
            method={paymentMethod}
            onMethodChange={setPaymentMethod}
            onSubmit={() => setStep('review')}
            onBack={() => setStep('billing')}
          />
        )}

        {/* Review & Confirm Step */}
        {step === 'review' && (
          <OrderReview
            items={items}
            billing={billingInfo}
            payment={paymentMethod}
            onConfirm={async () => {
              // Process order
              const order = await processOrder({
                items,
                billing: billingInfo,
                payment: paymentMethod,
              });
              router.push(`/orders/${order.id}/success`);
            }}
            onBack={() => setStep('payment')}
          />
        )}
      </div>
    </div>
  );
};
```

### 5. License Management Interface

```typescript
// License Dashboard
export const LicenseDashboard: React.FC = () => {
  const { data: licenses } = useQuery({
    queryKey: ['licenses'],
    queryFn: getUserLicenses,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">My Licenses</h2>
        <Link
          href="/marketplace"
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
        >
          Browse Marketplace
        </Link>
      </div>

      {/* License Cards */}
      <div className="grid gap-6">
        {licenses?.map((license) => (
          <LicenseCard key={license.id} license={license} />
        ))}
      </div>
    </div>
  );
};

// Individual License Card
export const LicenseCard: React.FC<{ license: License }> = ({ license }) => {
  const [showKey, setShowKey] = useState(false);
  const [isActivating, setIsActivating] = useState(false);

  const handleActivate = async () => {
    setIsActivating(true);
    try {
      await activateLicense(license.id);
      toast.success('License activated successfully');
      queryClient.invalidateQueries(['licenses']);
    } catch (error) {
      toast.error('Failed to activate license');
    } finally {
      setIsActivating(false);
    }
  };

  return (
    <div className="bg-white rounded-lg border p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {/* Product Info */}
          <div className="flex items-center gap-4 mb-4">
            <img
              src={license.product.icon}
              alt={license.product.name}
              className="w-12 h-12 rounded-lg"
            />
            <div>
              <h3 className="font-semibold text-lg">{license.product.name}</h3>
              <p className="text-sm text-gray-600">
                {license.product.vendor.name}
              </p>
            </div>
          </div>

          {/* License Details */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <span className="text-sm text-gray-600">Status</span>
              <div className="flex items-center gap-2 mt-1">
                <StatusBadge status={license.status} />
              </div>
            </div>

            <div>
              <span className="text-sm text-gray-600">Type</span>
              <div className="font-medium mt-1">{license.type}</div>
            </div>

            <div>
              <span className="text-sm text-gray-600">Expires</span>
              <div className="font-medium mt-1">
                {license.expiresAt
                  ? formatDate(license.expiresAt)
                  : 'Never'}
              </div>
            </div>
          </div>

          {/* License Key */}
          <div className="bg-gray-50 rounded p-3">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <span className="text-sm text-gray-600">License Key</span>
                <div className="font-mono text-sm mt-1">
                  {showKey
                    ? license.key
                    : '••••-••••-••••-••••'}
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowKey(!showKey)}
                  className="p-2 hover:bg-gray-200 rounded"
                >
                  {showKey ? <EyeOff /> : <Eye />}
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(license.key);
                    toast.success('License key copied');
                  }}
                  className="p-2 hover:bg-gray-200 rounded"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Usage Stats */}
          {license.usage && (
            <div className="mt-4 grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {license.usage.activations}
                </div>
                <div className="text-xs text-gray-600">Activations</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {license.usage.remaining}
                </div>
                <div className="text-xs text-gray-600">Remaining</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {license.usage.apiCalls}
                </div>
                <div className="text-xs text-gray-600">API Calls</div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="ml-6 space-y-2">
          {license.status === 'inactive' && (
            <button
              onClick={handleActivate}
              disabled={isActivating}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              {isActivating ? 'Activating...' : 'Activate'}
            </button>
          )}

          <Link
            href={`/licenses/${license.id}/manage`}
            className="block px-4 py-2 border rounded hover:bg-gray-50 text-center"
          >
            Manage
          </Link>

          <Link
            href={`/support/license/${license.id}`}
            className="block px-4 py-2 text-gray-600 hover:text-gray-900 text-center"
          >
            Get Support
          </Link>
        </div>
      </div>
    </div>
  );
};
```

### 6. Subscription Management

```typescript
// Subscription Management Dashboard
export const SubscriptionManager: React.FC = () => {
  const { data: subscriptions } = useQuery({
    queryKey: ['subscriptions'],
    queryFn: getUserSubscriptions,
  });

  return (
    <div className="space-y-6">
      {/* Active Subscriptions */}
      <div>
        <h3 className="text-xl font-semibold mb-4">Active Subscriptions</h3>
        <div className="grid gap-4">
          {subscriptions?.active.map((sub) => (
            <SubscriptionCard key={sub.id} subscription={sub} />
          ))}
        </div>
      </div>

      {/* Billing History */}
      <div>
        <h3 className="text-xl font-semibold mb-4">Billing History</h3>
        <BillingHistoryTable />
      </div>

      {/* Payment Methods */}
      <div>
        <h3 className="text-xl font-semibold mb-4">Payment Methods</h3>
        <PaymentMethodsManager />
      </div>
    </div>
  );
};

// Subscription Card Component
export const SubscriptionCard: React.FC<{
  subscription: Subscription;
}> = ({ subscription }) => {
  const [showCancelModal, setShowCancelModal] = useState(false);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);

  return (
    <>
      <div className="bg-white rounded-lg border p-6">
        <div className="flex items-start justify-between">
          <div>
            <h4 className="font-semibold text-lg mb-2">
              {subscription.product.name}
            </h4>
            <p className="text-gray-600 mb-4">
              {subscription.plan.name} Plan
            </p>

            <div className="grid grid-cols-4 gap-4">
              <div>
                <span className="text-sm text-gray-600">Price</span>
                <div className="font-medium">
                  ${subscription.plan.price}/{subscription.plan.interval}
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-600">Next billing</span>
                <div className="font-medium">
                  {formatDate(subscription.nextBillingDate)}
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-600">Status</span>
                <div>
                  <StatusBadge status={subscription.status} />
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-600">Auto-renew</span>
                <div>
                  <Switch
                    checked={subscription.autoRenew}
                    onCheckedChange={(checked) =>
                      updateAutoRenew(subscription.id, checked)
                    }
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <button
              onClick={() => setShowUpgradeModal(true)}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Upgrade Plan
            </button>
            <button
              onClick={() => setShowCancelModal(true)}
              className="px-4 py-2 border rounded hover:bg-gray-50"
            >
              Cancel Subscription
            </button>
          </div>
        </div>

        {/* Usage Metrics */}
        <div className="mt-6 pt-6 border-t">
          <h5 className="font-medium mb-3">Current Usage</h5>
          <div className="space-y-2">
            {subscription.usage.map((metric) => (
              <div key={metric.name} className="flex items-center gap-4">
                <span className="text-sm text-gray-600 w-32">
                  {metric.name}
                </span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-600 h-full rounded-full"
                    style={{
                      width: `${(metric.current / metric.limit) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-sm">
                  {metric.current} / {metric.limit}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cancel Modal */}
      {showCancelModal && (
        <CancelSubscriptionModal
          subscription={subscription}
          onClose={() => setShowCancelModal(false)}
        />
      )}

      {/* Upgrade Modal */}
      {showUpgradeModal && (
        <UpgradePlanModal
          subscription={subscription}
          onClose={() => setShowUpgradeModal(false)}
        />
      )}
    </>
  );
};
```

### 7. Revenue Analytics (Seller Dashboard)

```typescript
// Seller Revenue Dashboard
export const SellerDashboard: React.FC = () => {
  const [dateRange, setDateRange] = useState<DateRange>({
    start: subDays(new Date(), 30),
    end: new Date(),
  });

  const { data: analytics } = useQuery({
    queryKey: ['seller-analytics', dateRange],
    queryFn: () => getSellerAnalytics(dateRange),
  });

  return (
    <div className="space-y-6">
      {/* Revenue Overview */}
      <div className="grid grid-cols-4 gap-6">
        <RevenueCard
          title="Total Revenue"
          value={`$${formatNumber(analytics?.totalRevenue || 0)}`}
          change={analytics?.revenueChange}
          icon="dollar-sign"
        />
        <RevenueCard
          title="Active Subscriptions"
          value={formatNumber(analytics?.activeSubscriptions || 0)}
          change={analytics?.subscriptionChange}
          icon="users"
        />
        <RevenueCard
          title="Avg. Order Value"
          value={`$${analytics?.avgOrderValue?.toFixed(2) || '0'}`}
          change={analytics?.aovChange}
          icon="shopping-cart"
        />
        <RevenueCard
          title="Conversion Rate"
          value={`${analytics?.conversionRate?.toFixed(1) || '0'}%`}
          change={analytics?.conversionChange}
          icon="trending-up"
        />
      </div>

      {/* Revenue Chart */}
      <div className="bg-white rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Revenue Overview</h3>
          <DateRangePicker
            value={dateRange}
            onChange={setDateRange}
          />
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={analytics?.revenueTimeline}>
            <defs>
              <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip formatter={(value) => `$${value}`} />
            <Legend />
            <Area
              type="monotone"
              dataKey="revenue"
              stroke="#10b981"
              fillOpacity={1}
              fill="url(#colorRevenue)"
            />
            <Area
              type="monotone"
              dataKey="subscriptions"
              stroke="#3b82f6"
              fillOpacity={0.6}
              fill="#3b82f6"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Product Performance */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Product Performance</h3>
        <DataTable
          columns={[
            { key: 'name', label: 'Product', sortable: true },
            { key: 'sales', label: 'Sales', sortable: true },
            { key: 'revenue', label: 'Revenue', sortable: true },
            { key: 'conversionRate', label: 'Conversion', sortable: true },
            { key: 'rating', label: 'Rating', sortable: true },
            { key: 'trend', label: 'Trend' },
          ]}
          data={analytics?.products || []}
          renderCell={(key, value, row) => {
            if (key === 'revenue') return `$${formatNumber(value)}`;
            if (key === 'conversionRate') return `${value.toFixed(1)}%`;
            if (key === 'rating') return <RatingStars rating={value} />;
            if (key === 'trend') return <TrendIndicator value={value} />;
            return value;
          }}
        />
      </div>

      {/* Payout Information */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Upcoming Payouts</h3>
        <PayoutSchedule payouts={analytics?.payouts || []} />
      </div>
    </div>
  );
};
```

### 8. Enterprise Features

```typescript
// Enterprise Marketplace Section
export const EnterpriseMarketplace: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero */}
      <section className="bg-gradient-to-br from-gray-900 to-gray-800 text-white py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl">
            <h1 className="text-5xl font-bold mb-6">
              Enterprise Solutions
            </h1>
            <p className="text-xl mb-8 opacity-90">
              Custom climate intelligence solutions designed for your organization's unique needs
            </p>
            <div className="flex gap-4">
              <button className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700">
                Schedule Demo
              </button>
              <button className="px-6 py-3 border-2 border-white rounded-lg hover:bg-white hover:text-gray-900">
                Contact Sales
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Solutions Grid */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-12 text-center">
            Tailored Solutions for Every Industry
          </h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <EnterpriseSolutionCard
              title="White-Label Platform"
              description="Fully customizable climate intelligence platform with your branding"
              features={[
                'Custom branding',
                'Dedicated infrastructure',
                'API access',
                'Priority support',
              ]}
              startingPrice="$10,000"
            />

            <EnterpriseSolutionCard
              title="Custom Agent Development"
              description="Bespoke climate agents tailored to your specific requirements"
              features={[
                'Requirements analysis',
                'Custom development',
                'Integration support',
                'Ongoing maintenance',
              ]}
              startingPrice="$25,000"
            />

            <EnterpriseSolutionCard
              title="Managed Services"
              description="Full-service climate intelligence management for your organization"
              features={[
                '24/7 monitoring',
                'Dedicated team',
                'Compliance reporting',
                'Strategic consulting',
              ]}
              startingPrice="$5,000"
            />
          </div>
        </div>
      </section>

      {/* ROI Calculator */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold mb-8 text-center">
              Calculate Your ROI
            </h2>
            <ROICalculator />
          </div>
        </div>
      </section>

      {/* Case Studies */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-12 text-center">
            Success Stories
          </h2>
          <CaseStudyCarousel />
        </div>
      </section>
    </div>
  );
};
```

### 9. Performance Optimization

```yaml
Performance Targets:
  - Page Load: <2s (with code splitting)
  - Time to Interactive: <3s
  - Cart Operations: <100ms
  - Search Response: <200ms
  - Checkout Flow: <5s total

Optimization Strategies:
  - Image optimization with next/image
  - Lazy loading for product images
  - Virtual scrolling for large lists
  - Server-side rendering for SEO
  - Static generation for product pages
  - Edge caching with CDN
  - Bundle splitting by route

Payment Security:
  - PCI DSS compliance
  - SSL/TLS encryption
  - Tokenization with Stripe
  - Fraud detection
  - 3D Secure authentication
```

### 10. Timeline & Milestones

```yaml
Q1 2027: Foundation
  Month 1:
    - Product listing pages
    - Shopping cart implementation
    - Basic checkout flow
    - Stripe integration

  Month 2:
    - License management
    - Subscription handling
    - Seller dashboard
    - Revenue analytics

  Month 3:
    - Enterprise features
    - Advanced search
    - Review system
    - Beta testing

Q2 2027: Launch
  - 500+ products listed
  - Full payment processing
  - International support
  - Tax calculation
  - Public launch

Q3 2027: Scale
  - 2,500+ products
  - Marketplace API
  - Mobile app
  - AI recommendations
  - $5M GMV target
```