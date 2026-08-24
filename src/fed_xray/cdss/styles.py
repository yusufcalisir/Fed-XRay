"""
Fed-XRay Design System 2.0 & CSS Tokens
========================================
Implements modern clinical SaaS styling, HSL color tokens, glassmorphic cards,
and universal cross-device responsive grid layouts.
"""


def get_custom_css() -> str:
    """Return complete injected CSS stylesheet for Design System 2.0."""
    return """
    <style>
    /* ===== GOOGLE FONTS IMPORT ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== ROOT DESIGN TOKENS ===== */
    :root {
        --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-display: 'Plus Jakarta Sans', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;

        /* HSL Color System */
        --clr-bg-dark: #0B0F19;
        --clr-bg-card: rgba(15, 23, 42, 0.75);
        --clr-bg-card-light: rgba(255, 255, 255, 0.95);
        
        --clr-primary: #06B6D4;
        --clr-primary-glow: rgba(6, 182, 212, 0.25);
        --clr-accent: #6366F1;
        --clr-accent-glow: rgba(99, 102, 241, 0.25);

        --clr-benign: #10B981;
        --clr-benign-bg: rgba(16, 185, 129, 0.12);
        --clr-warning: #F59E0B;
        --clr-warning-bg: rgba(245, 158, 11, 0.12);
        --clr-malignant: #EF4444;
        --clr-malignant-bg: rgba(239, 68, 68, 0.12);

        --clr-text-main: #0F172A;
        --clr-text-muted: #64748B;
        --clr-border: rgba(226, 232, 240, 0.8);
        
        --radius-sm: 8px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --shadow-subtle: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
        --shadow-glow: 0 8px 30px rgba(6, 182, 212, 0.18);
    }

    /* Global Base Reset */
    html, body, [class*="css"] {
        font-family: var(--font-primary) !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Custom Header container */
    .fed-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-lg);
        padding: 2.25rem 2rem;
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.5);
    }

    .fed-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.22) 0%, rgba(99, 102, 241, 0.05) 50%, transparent 70%);
        pointer-events: none;
    }

    .fed-hero-title {
        font-family: var(--font-display) !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.2 !important;
    }

    .fed-hero-subtitle {
        font-size: 1rem !important;
        color: #94A3B8 !important;
        margin: 0 !important;
        font-weight: 400 !important;
        max-width: 800px;
        line-height: 1.5 !important;
    }

    /* ===== METRICS GRID ===== */
    .fed-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.75rem;
    }

    .fed-metric-card {
        background: #FFFFFF;
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-subtle);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .fed-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.08);
        border-color: rgba(6, 182, 212, 0.4);
    }

    .fed-metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--clr-primary), var(--clr-accent));
        border-radius: 4px 0 0 4px;
    }

    .fed-metric-label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--clr-text-muted) !important;
        margin-bottom: 0.35rem !important;
    }

    .fed-metric-value {
        font-family: var(--font-display) !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--clr-text-main) !important;
        line-height: 1 !important;
    }

    /* ===== STATUS BADGES ===== */
    .fed-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .fed-badge-benign {
        background: var(--clr-benign-bg);
        color: #065F46;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .fed-badge-warning {
        background: var(--clr-warning-bg);
        color: #92400E;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .fed-badge-malignant {
        background: var(--clr-malignant-bg);
        color: #991B1B;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .fed-badge-primary {
        background: rgba(6, 182, 212, 0.12);
        color: #0E7490;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    /* ===== GLASS CONTAINER & SECTION CARDS ===== */
    .fed-section-card {
        background: #FFFFFF;
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-subtle);
    }

    .fed-section-title {
        font-family: var(--font-display) !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-bottom: 0.5rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .fed-section-desc {
        font-size: 0.9rem !important;
        color: #64748B !important;
        margin-bottom: 1.25rem !important;
        line-height: 1.5 !important;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: #0F172A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5 {
        color: #FFFFFF !important;
        font-family: var(--font-display) !important;
    }

    /* Modern Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #06B6D4 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #0891B2 0%, #0369A1 100%) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ===== STREAMLIT TAB OVERHAUL ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #F1F5F9;
        padding: 0.35rem;
        border-radius: var(--radius-md);
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 0.6rem 1.25rem;
        font-family: var(--font-display) !important;
        font-weight: 600;
        font-size: 0.9rem;
        color: #64748B;
        border: none !important;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
    }

    /* ===== UNIVERSAL RESPONSIVE MATRIX ===== */
    @media (max-width: 768px) {
        .fed-hero {
            padding: 1.5rem 1.25rem;
        }
        .fed-hero-title {
            font-size: 1.5rem !important;
        }
        .fed-metrics-grid {
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }
        .fed-metric-value {
            font-size: 1.35rem !important;
        }
        .fed-section-card {
            padding: 1rem;
        }
    }

    @media (min-width: 1440px) {
        .fed-hero-title {
            font-size: 2.5rem !important;
        }
        .fed-metric-value {
            font-size: 2rem !important;
        }
    }
    </style>
    """
