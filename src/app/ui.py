import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body {
            height: 100%;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stDeployButton"] {display: none;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        section[data-testid="stSidebarNav"] {display: none;}
        .stApp {
            background: radial-gradient(circle at top, #111827 0%, #0b0f1a 45%, #05060b 100%);
            color: #f8fafc;
            height: 100%;
        }
        .block-container {
            padding: 0 32px;
            margin: 0 auto;
            max-width: 1100px;
        }
        @media (max-width: 720px) {
            .block-container {
                padding: 0 20px;
            }
        }
        .nav-shell {
            position: fixed;
            top: 0;
            left: 0;
            transform: none;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 16px 22px;
            border-radius: 0;
            background: rgba(10, 14, 23, 0.75);
            box-shadow: 0 20px 50px rgba(2, 6, 23, 0.55);
            backdrop-filter: blur(10px);
            z-index: 50;
        }
        .nav-shell .stButton > button {
            height: 42px;
        }
        .nav-logo {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: #ffffff;
        }
        .nav-links {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .nav-links a {
            text-decoration: none;
            color: #e0e7ff;
            padding: 10px 18px;
            border-radius: 999px;
            border: 1px solid rgba(99, 102, 241, 0.35);
            background: rgba(15, 23, 42, 0.5);
            font-size: 14px;
        }
        .nav-links a.active {
            border-color: rgba(129, 140, 248, 0.9);
            box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
        }
        .nav-login {
            text-decoration: none;
            color: #0b0f1a !important;
            padding: 10px 22px;
            border-radius: 14px;
            background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
            box-shadow: 0 16px 32px rgba(79, 70, 229, 0.45);
            font-weight: 600;
            font-size: 14px;
        }
        .nav-login:visited,
        .nav-login:hover,
        .nav-login:active {
            color: #0b0f1a !important;
        }
        .app-hero {
            background: #0b0f1a;
            color: #ffffff;
            padding: 0;
            border-radius: 0;
            box-shadow: 0 30px 80px rgba(14, 18, 30, 0.75);
            margin-bottom: 0;
            position: relative;
            overflow: hidden;
            min-height: 100vh;
            display: flex;
            align-items: center;
            width: 100vw;
            margin-left: 50%;
            transform: translateX(-50%);
        }
        .app-hero--compact {
            min-height: 60vh;
        }
        .enterprise-hero {
            background: linear-gradient(180deg, #0b0f1a 0%, #0f172a 100%);
            box-shadow: none;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }
        .enterprise-hero .hero-glow,
        .enterprise-hero .hero-gradient {
            display: none;
        }
        .enterprise-hero h1 {
            font-size: 34px;
        }
        .enterprise-hero p {
            font-size: 16px;
            color: #cbd5f5;
        }
        .enterprise-panel {
            background: #f8fafc;
            color: #0f172a;
            border-radius: 20px;
            padding: 28px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.08);
            margin-top: 30px;
        }
        .enterprise-panel h3 {
            margin-top: 0;
            color: #0f172a;
        }
        .enterprise-panel label, 
        .enterprise-panel p, 
        .enterprise-panel .stMarkdown {
            color: #0f172a;
        }
        .enterprise-panel .stSlider [data-baseweb="slider"] {
            margin-top: 12px;
        }
        .enterprise-panel .stButton > button {
            background: #0f172a;
            color: #ffffff;
            border-radius: 12px;
        }
        .stack-card--enterprise {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.35);
            border-radius: 16px;
        }
        .rec-card--enterprise {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.3);
            border-radius: 14px;
        }
        
        .hero-inner {
            width: calc(100% - 96px);
            margin: 0 auto;
            padding: 120px 0 60px 0;
        }
        .hero-gradient {
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 15% 25%, rgba(99, 102, 241, 0.35), transparent 55%),
                        radial-gradient(circle at 85% 15%, rgba(14, 165, 233, 0.25), transparent 45%),
                        radial-gradient(circle at 80% 85%, rgba(236, 72, 153, 0.25), transparent 45%);
            pointer-events: none;
            animation: none;
        }
        @keyframes sweepGlow {
            0% { transform: translateX(-6%); opacity: 0.7; }
            50% { transform: translateX(6%); opacity: 1; }
            100% { transform: translateX(-6%); opacity: 0.7; }
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
            align-items: center;
        }
        .hero-actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 18px;
        }
        .app-hero h1 {
            font-size: 46px;
            margin-bottom: 10px;
        }
        .app-hero p {
            font-size: 18px;
            opacity: 0.9;
            max-width: 560px;
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            font-size: 12px;
            margin-right: 8px;
        }
        .hero-glow {
            position: absolute;
            width: 260px;
            height: 260px;
            background: rgba(59, 130, 246, 0.35);
            filter: blur(40px);
            top: -60px;
            right: -40px;
            animation: pulseGlow 6s ease-in-out infinite;
        }
        @keyframes pulseGlow {
            0%, 100% { transform: scale(1); opacity: 0.6; }
            50% { transform: scale(1.15); opacity: 0.9; }
        }
        .section-title {
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 14px;
            color: #f8fafc;
        }
        .section-subtitle {
            color: #cbd5f5;
            font-size: 15px;
            max-width: 680px;
        }
        .section-header {
            margin-bottom: 26px;
        }
        .section-container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 60px 32px;
        }
        .section-eyebrow {
            font-size: 12px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: rgba(148, 163, 184, 0.9);
            margin-bottom: 8px;
        }
        .section-title-lg {
            font-size: 30px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #f8fafc;
        }
        .section-title-lg,
        .section-subtitle {
            padding-left: 4px;
        }
        .section-divider {
            width: 64px;
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #0ea5e9);
            border-radius: 999px;
            margin-top: 14px;
        }
        .section-spacer {
            height: 40px;
        }
        .site-footer {
            background: rgba(10, 14, 23, 0.9);
            border-top: 1px solid rgba(99, 102, 241, 0.2);
            padding: 32px 0;
            margin-top: 40px;
            width: 100vw;
            margin-left: 50%;
            transform: translateX(-50%);
        }
        .footer-inner {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #94a3b8;
            font-size: 13px;
        }
        .footer-links {
            display: flex;
            gap: 16px;
        }
        .footer-links a {
            color: #cbd5f5;
            text-decoration: none;
        }
        .footer-links a:hover {
            color: #ffffff;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 18px;
            margin-top: 16px;
            padding: 0 4px;
        }
        .feature-card-pro {
            background: linear-gradient(150deg, rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.6));
            border-radius: 22px;
            padding: 26px;
            border: 1px solid rgba(99, 102, 241, 0.28);
            box-shadow: 0 20px 50px rgba(12, 16, 29, 0.5);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        .feature-card-pro:hover {
            transform: translateY(-8px);
            box-shadow: 0 28px 60px rgba(12, 16, 29, 0.6);
        }
        .feature-card-pro::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.18), transparent 60%);
            opacity: 0.7;
            pointer-events: none;
        }
        .feature-card-pro::before {
            content: "";
            position: absolute;
            inset: -40%;
            background: radial-gradient(circle, rgba(14, 165, 233, 0.22), transparent 55%);
            opacity: 0.4;
            filter: blur(20px);
            pointer-events: none;
        }
        .feature-icon {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            font-size: 20px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(14, 165, 233, 0.35));
            margin-bottom: 14px;
        }
        .feature-card {
            background: rgba(15, 23, 42, 0.8);
            border-radius: 18px;
            padding: 20px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        .feature-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.35);
        }
        .feature-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.15), transparent 60%);
            opacity: 0.6;
            pointer-events: none;
        }
        .stack-card {
            background: rgba(15, 23, 42, 0.75);
            border-radius: 20px;
            padding: 26px;
            border: 1px solid rgba(99, 102, 241, 0.35);
            box-shadow: 0 22px 50px rgba(15, 23, 42, 0.45);
            width: 100%;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(99, 102, 241, 0.4);
            background: rgba(30, 41, 59, 0.6);
            font-size: 12px;
            color: #e0e7ff;
        }
        .scroll-cue {
            margin-top: 28px;
            display: inline-flex;
            gap: 8px;
            align-items: center;
            font-size: 12px;
            color: #cbd5f5;
            opacity: 0.7;
        }
        .scroll-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #a5b4fc;
            animation: scrollPulse 1.4s ease-in-out infinite;
        }
        @keyframes scrollPulse {
            0%, 100% { transform: translateY(0); opacity: 0.6; }
            50% { transform: translateY(6px); opacity: 1; }
        }
        .rec-card-link {
            text-decoration: none;
            display: block;
            margin-bottom: 18px;
        }
        .rec-card-link:hover {
            text-decoration: none;
        }
        .rec-card {
            background: #0b1220;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
            margin-bottom: 18px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .rec-card--clickable {
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .rec-card--clickable::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(14, 165, 233, 0.1));
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .rec-card--clickable:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
            border-color: rgba(99, 102, 241, 0.4);
        }
        .rec-card--clickable:hover::before {
            opacity: 1;
        }
        .rec-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        .rec-card h3 {
            margin-top: 0;
            margin-bottom: 0;
            color: #f8fafc;
            font-size: 20px;
            font-weight: 600;
            flex: 1;
            transition: color 0.3s ease;
        }
        .rec-card--clickable:hover h3 {
            color: #a5b4fc;
        }
        .rec-card-arrow {
            color: #6366f1;
            font-size: 24px;
            font-weight: 300;
            opacity: 0;
            transform: translateX(-8px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .rec-card--clickable:hover .rec-card-arrow {
            opacity: 1;
            transform: translateX(0);
        }
        .rec-card-description {
            color: #cbd5e1;
            margin-bottom: 16px;
            line-height: 1.6;
            font-size: 14px;
        }
        .rec-card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
        }
        .rec-meta {
            color: #94a3b8;
            font-size: 12px;
        }
        .rec-score {
            color: #a5b4fc;
            font-weight: 500;
        }
        .nav-row {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 16px;
        }
        .footer-note {
            color: #64748b;
            font-size: 12px;
            margin-top: 28px;
        }
        .section-shell {
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .snap-section {
            scroll-snap-align: start;
        }
        .stApp {
            scroll-snap-type: y proximity;
        }
        @media (max-width: 1024px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero-inner {
                padding: 110px 0 50px 0;
            }
        }
        @media (max-width: 720px) {
            .hero-inner {
                width: calc(100% - 40px);
                padding: 100px 0 40px 0;
            }
            .nav-shell {
                width: calc(100% - 24px);
                top: 12px;
            }
        }
        .logo-row {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            opacity: 0.7;
            font-size: 12px;
        }
        .testimonial-card {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 18px;
            padding: 20px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
            position: relative;
            overflow: hidden;
        }
        .testimonial-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 10% 10%, rgba(236, 72, 153, 0.16), transparent 55%);
            opacity: 0.5;
            pointer-events: none;
        }
        .auth-shell {
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 80px 24px;
        }
        .auth-card {
            width: min(460px, 100%);
            background: rgba(10, 14, 23, 0.85);
            border-radius: 20px;
            border: 1px solid rgba(99, 102, 241, 0.35);
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.6);
            padding: 36px;
        }
        .auth-title {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .auth-subtitle {
            color: #cbd5f5;
            font-size: 14px;
            margin-bottom: 24px;
        }
        .auth-center {
            text-align: center;
        }
        .auth-header {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 6px;
            margin-bottom: 18px;
        }
        .auth-brand {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.4px;
        }
        .auth-back {
            font-size: 12px;
            color: #a5b4fc;
            text-decoration: none;
        }
        .auth-back:hover {
            color: #c7d2fe;
        }
        .auth-link {
            color: #93c5fd;
            text-decoration: none;
        }
        .auth-link:hover {
            color: #bfdbfe;
        }
        div[data-testid="stForm"] {
            background: rgba(10, 14, 23, 0.85);
            border-radius: 20px;
            border: 1px solid rgba(99, 102, 241, 0.35);
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.6);
            padding: 28px;
        }
        div[data-testid="stForm"] button {
            width: 100%;
        }
        .stTextInput [data-baseweb="input"] {
            position: relative;
            align-items: center;
        }
        .stTextInput button[aria-label="View password"],
        .stTextInput button[aria-label="Hide password"] {
            display: none !important;
        }
        button[kind="secondary"] {
            border-radius: 999px;
            border: 1px solid rgba(99, 102, 241, 0.45);
            background: rgba(15, 23, 42, 0.5);
            color: #e0e7ff;
        }
        button[kind="secondary"]:hover {
            border-color: rgba(129, 140, 248, 0.9);
            color: #ffffff;
        }
        button[kind="primary"] {
            border-radius: 14px;
            border: none;
            background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
            box-shadow: 0 16px 32px rgba(79, 70, 229, 0.45);
            color: #ffffff;
        }
        button[kind="primary"]:hover {
            filter: brightness(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_nav(active_route: str) -> None:
    active = {
        "home": "active" if active_route == "home" else "",
        "explore": "active" if active_route == "explore" else "",
        "signup": "active" if active_route == "signup" else "",
    }
    st.markdown(
        f"""
        <div class="nav-shell">
            <div class="nav-logo">Skillens</div>
            <div class="nav-links">
                <a class="{active['home']}" href="?page=home" target="_self">Home</a>
                <a class="{active['explore']}" href="?page=explore" target="_self">Explore</a>
            </div>
            <a class="nav-login" href="?page=login" target="_self">Guest session</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

