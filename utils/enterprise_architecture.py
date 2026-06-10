"""
Fed-XRay: Enterprise Security & Governance Control Plane Engine
==============================================================
Provides simulated implementations and dashboards for:
- Cloud & Multi Cloud Deployments
- Zero Trust Network Architecture (ZTNA)
- Web Application Firewall (WAF)
- Security Operations Center (SOC) & SIEM log streams
- Privileged Access Management (PAM)
- ITIL v4 & COBIT 2019 Compliance Mapping
- Executive Audit Report PDF generation
"""

import os
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import streamlit as st
from fpdf import FPDF

# ============================================================================
# STATE INITIALIZATION
# ============================================================================

def init_enterprise_state():
    """Initialize state variables for security and enterprise features."""
    if 'siem_logs' not in st.session_state:
        st.session_state.siem_logs = [
            {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": "GATEWAY", "event": "Zero Trust Gateway initialized.", "level": "INFO"},
            {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": "SIEM", "event": "Ingestion channels online.", "level": "INFO"},
            {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": "WAF", "event": "Web Application Firewall activated with OWASP Core Rules.", "level": "INFO"}
        ]
    if 'zt_policies' not in st.session_state:
        st.session_state.zt_policies = {
            "mtls": True,
            "device_compliance": True,
            "geo_fencing": False,
            "session_expiry": True
        }
    if 'pam_session' not in st.session_state:
        st.session_state.pam_session = {
            "is_elevated": False,
            "elevated_until": 0,
            "request_mfa": False,
            "mfa_token": "",
            "mfa_attempts": 0,
            "audit_trail": [
                "System initialized: default auditor session opened."
            ]
        }
    if 'quarantined_nodes' not in st.session_state:
        st.session_state.quarantined_nodes = set()
    if 'simulated_incidents' not in st.session_state:
        st.session_state.simulated_incidents = []
    if 'multi_cloud_latency' not in st.session_state:
        st.session_state.multi_cloud_latency = {
            "AWS-East (Server)": 0,
            "AWS-West (Hospital 1)": 45,
            "Azure-Europe (Hospital 2)": 110,
            "GCP-Asia (Hospital 3)": 230,
            "Private Cloud (Hospital 4)": 15
        }
    if 'waf_rule_hits' not in st.session_state:
        st.session_state.waf_rule_hits = {
            "SQL Injection Blocked": 0,
            "DDoS Filtered Requests": 0,
            "Malicious Weight Outliers": 0
        }

# ============================================================================
# LOGS GENERATOR Helper
# ============================================================================

def log_event(source: str, event: str, level: str = "INFO"):
    """Append a security event log to the SIEM log list."""
    if 'siem_logs' not in st.session_state:
        st.session_state.siem_logs = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.siem_logs.insert(0, {
        "time": timestamp,
        "source": source,
        "event": event,
        "level": level
    })
    # Keep last 50 logs
    st.session_state.siem_logs = st.session_state.siem_logs[:50]

# ============================================================================
# COMPLIANCE REPORT GENERATOR (PDF)
# ============================================================================

def generate_governance_pdf() -> bytes:
    """Generate a clean enterprise compliance report for COBIT & ITIL auditing."""
    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Header
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 10, 'Fed-XRay Enterprise Security & Governance Audit', 0, 1, 'C')
    
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f'Audit Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Classification: CONFIDENTIAL', 0, 1, 'C')
    pdf.line(15, 26, 195, 26)
    pdf.ln(5)
    
    # Section 1: Executive Summary
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 6, '1. EXECUTIVE SUMMARY & SECURITY CONTROLS', 0, 1)
    
    pdf.set_font('Arial', '', 9.5)
    pdf.set_text_color(0, 0, 0)
    summary_text = (
        "This report documents the security governance profile of the Fed-XRay Federated Learning Platform. "
        "Unlike centralized AI networks, Fed-XRay operates under strict data minimization rules: patient data "
        "remains localized, and only encrypted model updates are exchanged. Security controls are mapped against "
        "industry standards (COBIT 2019 and ITIL v4) to ensure high operational resilience, data privacy, and active risk mitigation."
    )
    pdf.multi_cell(0, 4.5, summary_text)
    pdf.ln(4)
    
    # Section 2: Zero Trust Configuration
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 6, '2. ZERO TRUST AND NETWORK CONTROLS', 0, 1)
    
    pdf.set_font('Arial', '', 9.5)
    zt_status = st.session_state.zt_policies
    pdf.cell(90, 5, f"- Mutual TLS Authentication (mTLS): {'ENABLED' if zt_status['mtls'] else 'DISABLED'}", 0, 0)
    pdf.cell(90, 5, f"- Device Posture Compliance Checks: {'ENABLED' if zt_status['device_compliance'] else 'DISABLED'}", 0, 1)
    pdf.cell(90, 5, f"- Geo-Fencing & IP Restrictions: {'ENABLED' if zt_status['geo_fencing'] else 'DISABLED'}", 0, 0)
    pdf.cell(90, 5, f"- Admin Session Expiry Enforcement: {'ENABLED' if zt_status['session_expiry'] else 'DISABLED'}", 0, 1)
    
    # WAF stats
    waf_hits = st.session_state.waf_rule_hits
    pdf.cell(0, 5, f"- WAF Event Blocks: {waf_hits['SQL Injection Blocked']} SQLi | {waf_hits['DDoS Filtered Requests']} DDoS attempts", 0, 1)
    pdf.ln(4)
    
    # Section 3: COBIT 2019 Framework Mapping
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 6, '3. COBIT 2019 GOVERNANCE ALIGNMENT', 0, 1)
    
    pdf.set_font('Arial', 'B', 8.5)
    pdf.set_fill_color(240, 244, 248)
    pdf.cell(45, 5.5, 'Domain Metric', 1, 0, 'C', True)
    pdf.cell(100, 5.5, 'Fed-XRay Control Mapping', 1, 0, 'C', True)
    pdf.cell(35, 5.5, 'Audit Status', 1, 1, 'C', True)
    
    pdf.set_font('Arial', '', 8.5)
    cobit_mappings = [
        ("EDM03 (Risk Optimization)", "Server-side outlier detection automatically flags and filters malicious gradient updates.", "Optimized"),
        ("APO12 (Managed Risk)", "Continuous threat model evaluation, active WAF shield logs, and automated alerting.", "Managed"),
        ("APO14 (Managed Data)", "Strict data localization. Models are serialized locally; no raw clinical data is transmitted.", "Compliant"),
        ("DSS05 (Managed Security Services)", "mTLS device posture evaluation, secure JWT authentication, and JIT privileges.", "Secured"),
        ("MEA01 (Monitor, Evaluate & Assess)", "Real-time SIEM activity log streams and integrated SOC incident console dashboards.", "Verified")
    ]
    
    for row in cobit_mappings:
        pdf.cell(45, 5.5, row[0], 1, 0, 'L')
        pdf.cell(100, 5.5, row[1], 1, 0, 'L')
        pdf.cell(35, 5.5, row[2], 1, 1, 'C')
    pdf.ln(4)
    
    # Section 4: ITIL v4 Service Management Alignment
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 6, '4. ITIL V4 SERVICE PRACTICE ALIGNMENT', 0, 1)
    
    pdf.set_font('Arial', '', 9.5)
    itil_mappings = [
        ("Information Security Management", "Ensures data confidentiality (zero raw sharing), integrity (validation-shield against label-flipping attacks), and availability of local training nodes."),
        ("Incident Management", "The SOC detects malicious model updates in real-time, triggers alerts, and places offending hospital clients in quarantine."),
        ("Change Enablement", "Model iterations are managed via strict communication rounds, verifying weights on hold-out global validations prior to central integration.")
    ]
    for title, desc in itil_mappings:
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 4.5, f"- {title}:", 0, 1)
        pdf.set_font('Arial', '', 8.5)
        pdf.multi_cell(0, 4, desc)
        pdf.ln(1)
        
    pdf.ln(3)
    
    # Footer Section
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 4, 'Fed-XRay Governance Report. Prepared automatically by the Enterprise Architecture Engine.', 0, 1, 'C')
    pdf.cell(0, 4, 'Unauthorized copying is strictly prohibited. Information Security Compliance Officer approved.', 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# ============================================================================
# COMPONENT RENDERERS
# ============================================================================

def render_cloud_section():
    """Render the Cloud & Multi Cloud Control Plane."""
    st.markdown('<h3 style="color:#1a365d;">☁️ Cloud & Multi-Cloud Control Plane</h3>', unsafe_allow_html=True)
    st.markdown(
        "Demonstrate the **Multi-Cloud Deployment** model where hospitals (client nodes) run on heterogeneous clouds, "
        "orchestrated by a centralized aggregation cluster."
    )
    
    # Redundancy controls
    col_control, col_net = st.columns([1, 1.5])
    
    with col_control:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>⚙️ Network Orchestration</h4>", unsafe_allow_html=True)
        failover_active = st.checkbox("Enable Multi-Cloud Traffic Failover", value=True)
        encrypt_channel = st.checkbox("Force TLS 1.3 Encryption", value=True)
        
        st.markdown("---")
        st.markdown("**Cloud Load Balancer Policy:**")
        lb_policy = st.selectbox("Routing Algorithm", ["Lowest Latency First", "Geographic Proximity", "Resource Availability"])
        
        # Simulated action
        if st.button("⚡ Trigger Route Optimization Scan", key="btn_optim"):
            # Randomize latencies slightly
            st.session_state.multi_cloud_latency["AWS-West (Hospital 1)"] = random.randint(30, 60)
            st.session_state.multi_cloud_latency["Azure-Europe (Hospital 2)"] = random.randint(90, 130)
            st.session_state.multi_cloud_latency["GCP-Asia (Hospital 3)"] = random.randint(200, 260)
            st.session_state.multi_cloud_latency["Private Cloud (Hospital 4)"] = random.randint(10, 25)
            log_event("CLOUD-LB", f"Optimized connections using '{lb_policy}' logic. TLS status: {'Enforced' if encrypt_channel else 'Permissive'}.", "INFO")
            st.success("✅ Optimized connections across multi-cloud network!")
            st.rerun()
            
    with col_net:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>📡 Active Node Registry</h4>", unsafe_allow_html=True)
        
        # Render clean cards for clouds
        nodes = [
            {"name": "Central Server (AWS EKS)", "cloud": "AWS (Amazon Web Services)", "region": "us-east-1 (N. Virginia)", "ip": "10.0.1.15", "latency": "0 ms (Local)"},
            {"name": "Hospital 1 (AWS Node)", "cloud": "AWS (Amazon Web Services)", "region": "us-west-2 (Oregon)", "ip": "34.210.45.18", "latency": f"{st.session_state.multi_cloud_latency['AWS-West (Hospital 1)']} ms"},
            {"name": "Hospital 2 (Azure Node)", "cloud": "Azure (Microsoft Cloud)", "region": "westeurope (Amsterdam)", "ip": "13.69.102.24", "latency": f"{st.session_state.multi_cloud_latency['Azure-Europe (Hospital 2)']} ms"},
            {"name": "Hospital 3 (GCP Node)", "cloud": "GCP (Google Cloud)", "region": "asia-east1 (Taiwan)", "ip": "35.220.12.87", "latency": f"{st.session_state.multi_cloud_latency['GCP-Asia (Hospital 3)']} ms"},
            {"name": "Hospital 4 (On-Premises)", "cloud": "Private OpenStack", "region": "Clinical On-Prem (Istanbul)", "ip": "192.168.4.120", "latency": f"{st.session_state.multi_cloud_latency['Private Cloud (Hospital 4)']} ms"}
        ]
        
        for n in nodes:
            # Latency Badge color
            lat_str = n['latency']
            if 'Local' in lat_str:
                lat_badge = '<span style="background:#e2e8f0; color:#4a5568; padding:3px 8px; border-radius:10px; font-size:0.75rem;">HOST</span>'
            elif int(lat_str.split()[0]) < 30:
                lat_badge = f'<span style="background:#c6f6d5; color:#22543d; padding:3px 8px; border-radius:10px; font-size:0.75rem; font-weight:600;">{lat_str}</span>'
            elif int(lat_str.split()[0]) < 120:
                lat_badge = f'<span style="background:#feebc8; color:#744210; padding:3px 8px; border-radius:10px; font-size:0.75rem; font-weight:600;">{lat_str}</span>'
            else:
                lat_badge = f'<span style="background:#fed7d7; color:#742a2a; padding:3px 8px; border-radius:10px; font-size:0.75rem; font-weight:600;">{lat_str}</span>'

            # Quarantine status check
            q_status = ""
            h_id_guess = None
            if "Hospital 1" in n['name']: h_id_guess = 0
            elif "Hospital 2" in n['name']: h_id_guess = 1
            elif "Hospital 3" in n['name']: h_id_guess = 2
            elif "Hospital 4" in n['name']: h_id_guess = 3
            
            if h_id_guess is not None and h_id_guess in st.session_state.quarantined_nodes:
                q_status = ' <span style="background:#fed7d7; color:#e53e3e; border:1px solid #c53030; padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:700; margin-left:10px;">QUARANTINED</span>'
            
            st.markdown(f"""
            <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem; display:flex; justify-content:between; align-items:center;">
                <div style="flex-grow:1;">
                    <strong style="font-size:0.95rem; color:#1a202c;">{n['name']}</strong>{q_status}<br>
                    <span style="font-size:0.75rem; color:#718096;">☁️ Provider: {n['cloud']} | Region: {n['region']} | IP: {n['ip']}</span>
                </div>
                <div style="text-align:right;">
                    {lat_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_zero_trust_section():
    """Render the Zero Trust Network Access (ZTNA) policies."""
    st.markdown('<h3 style="color:#1a365d;">🛡️ Zero Trust Network Architecture</h3>', unsafe_allow_html=True)
    st.markdown(
        "Configure and audit compliance under a strict **Zero Trust (Never Trust, Always Verify)** security stance. "
        "Hospitals must authenticate continuously before they can retrieve global parameters or push local weights."
    )
    
    col_policies, col_audit = st.columns([1, 1.3])
    
    with col_policies:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>🛠️ Policy Enforcement Console</h4>", unsafe_allow_html=True)
        
        # Bind toggles to session state
        mtls = st.toggle("Require Mutual TLS (mTLS v1.3)", value=st.session_state.zt_policies["mtls"])
        posture = st.toggle("Verify Client Device Posture (OS Patches & Firewall)", value=st.session_state.zt_policies["device_compliance"])
        geofence = st.toggle("Enforce Geographic Border Fencing", value=st.session_state.zt_policies["geo_fencing"])
        session = st.toggle("Enforce 15-Minute Token Expiry", value=st.session_state.zt_policies["session_expiry"])
        
        # Save policies
        if (mtls != st.session_state.zt_policies["mtls"] or 
            posture != st.session_state.zt_policies["device_compliance"] or 
            geofence != st.session_state.zt_policies["geo_fencing"] or 
            session != st.session_state.zt_policies["session_expiry"]):
            
            st.session_state.zt_policies["mtls"] = mtls
            st.session_state.zt_policies["device_compliance"] = posture
            st.session_state.zt_policies["geo_fencing"] = geofence
            st.session_state.zt_policies["session_expiry"] = session
            
            log_event("ZERO-TRUST-POLICY", f"Security policy updated: mTLS={mtls}, Posture={posture}, Geo-Fence={geofence}, Expiry={session}.", "WARNING")
            st.success("✅ Zero Trust Policies modified successfully.")
            st.rerun()
            
    with col_audit:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>🔒 Client Verification Audit</h4>", unsafe_allow_html=True)
        
        # Render a structured list showing each hospital's access authentication logs
        nodes = ["Hospital 1", "Hospital 2", "Hospital 3", "Hospital 4"]
        
        for idx, node in enumerate(nodes):
            checks = []
            authenticated = True
            
            # Check 1: mTLS
            if st.session_state.zt_policies["mtls"]:
                checks.append("🔒 mTLS: OK")
            else:
                checks.append("⚠️ mTLS: Skipped")
                
            # Check 2: Posture
            if st.session_state.zt_policies["device_compliance"]:
                # Let's say Hospital 3 is compromised or failing compliance if attack is active
                is_compromised = (st.sidebar.checkbox("⚠️ Simulate Attack (Hospital #3)", key="mock_att_zt_dummy", value=False) or idx == 2)
                if is_compromised and idx == 2:
                    checks.append("❌ OS: Outdated Patches")
                    authenticated = False
                else:
                    checks.append("💻 OS: Secure")
            else:
                checks.append("⚠️ OS: Unchecked")
                
            # Check 3: Geo
            if st.session_state.zt_policies["geo_fencing"]:
                if idx == 2:  # Hospital 3 is in Asia (Taiwan)
                    checks.append("❌ GEO: Foreign Region Blocked")
                    authenticated = False
                else:
                    checks.append("🗺️ GEO: Whitelisted")
            
            auth_status = (
                '<span style="background:#c6f6d5; color:#22543d; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">AUTHORIZED</span>' 
                if authenticated else 
                '<span style="background:#fed7d7; color:#742a2a; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">BLOCKED</span>'
            )
            
            checks_rendered = " | ".join(checks)
            st.markdown(f"""
            <div style="background:#f8fafc; border:1px solid #edf2f7; border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:0.4rem; display:flex; justify-content:between; align-items:center;">
                <div>
                    <span style="font-weight:600; color:#2d3748;">{node} Access Check</span><br>
                    <code style="font-size:0.75rem; color:#4a5568;">{checks_rendered}</code>
                </div>
                <div>
                    {auth_status}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_waf_section():
    """Render the WAF Section."""
    st.markdown('<h3 style="color:#1a365d;">🔥 Web Application Firewall (WAF)</h3>', unsafe_allow_html=True)
    st.markdown(
        "Demonstrate active filtering of HTTP payloads to the central server aggregator. "
        "The WAF mitigates standard exploit attempts (SQL Injection, XSS) and API volumetric abuse."
    )
    
    col_rules, col_attack = st.columns([1, 1.2])
    
    with col_rules:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>📋 Active WAF Rules</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **WAF-101:** SQL Injection Filter (`UNION`, `SELECT`, `--` blocks)
        - **WAF-102:** Cross-Site Scripting (XSS) Shield (`<script>` block)
        - **WAF-103:** Federated Weight Shape Verification (Ensures parameter shape matches exactly)
        - **WAF-104:** Rate-limiting Gate (Maximum 20 connection requests/minute per Client IP)
        """)
        
        # Show rule counters
        st.markdown("**Blocked Attack Logs:**")
        hits = st.session_state.waf_rule_hits
        st.info(f"🛡️ **Mitigations:** {hits['SQL Injection Blocked']} SQLi | {hits['DDoS Filtered Requests']} Rate Limit Blocks")
        
    with col_attack:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>💥 Attack Simulation Playground</h4>", unsafe_allow_html=True)
        st.write("Trigger simulated payloads to verify WAF rule block triggers:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 Simulate SQL Injection", key="btn_sqli"):
                st.session_state.waf_rule_hits["SQL Injection Blocked"] += 1
                log_event("WAF", "Rule WAF-101 Triggered: Blocked SQLi Payload from IP 198.51.100.42. Pattern matches: 'UNION SELECT'.", "CRITICAL")
                st.warning("🚨 WAF Action: SQLi blocked. Request terminated.")
                st.rerun()
                
        with col2:
            if st.button("⚡ Simulate DDoS Volumetric Request", key="btn_ddos"):
                st.session_state.waf_rule_hits["DDoS Filtered Requests"] += 1
                log_event("WAF", "Rule WAF-104 Triggered: Rate limits exceeded on Host 2 IP. Temporarily throttling client connection.", "WARNING")
                st.warning("🚨 WAF Action: Rate limiting applied to Node 2.")
                st.rerun()


def render_soc_siem_section():
    """Render SIEM log stream and SOC quarantine center."""
    st.markdown('<h3 style="color:#1a365d;">🚨 SOC (Security Operations Center) & SIEM</h3>', unsafe_allow_html=True)
    st.markdown(
        "Observe the SIEM event streaming platform which collects logs across the WAF, Zero Trust Gateway, "
        "and Federated aggregation cycles. The SOC Incident panel enables containment strategies (like node isolation)."
    )
    
    col_incidents, col_siem = st.columns([1, 1.2])
    
    with col_incidents:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>🚔 SOC Incident Response</h4>", unsafe_allow_html=True)
        
        # Check if Hospital 3 is compromised (attack simulated in sidebar)
        is_attack_active = st.session_state.get("start_training", False) and st.session_state.get("simulate_attack", False)
        # Or if the user manual toggled the sidebar attack checkbox
        if 'simulate_attack' in st.session_state and st.session_state.simulate_attack:
            is_attack_active = True
            
        if is_attack_active:
            st.markdown("""
            <div style="background:#fff5f5; border:1px solid #feb2b2; padding:1rem; border-radius:8px; margin-bottom:1rem;">
                <p style="color:#c53030 !important; font-weight:700; margin:0 0 0.5rem 0; font-size:0.95rem;">🚨 ALERT: Poisoned Update Attempted</p>
                <p style="font-size:0.8rem; color:#742a2a !important; margin:0 0 0.5rem 0;">
                    Node <strong>Hospital 3</strong> is broadcasting malicious weight variations (Label Flipping Pattern detected).
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action controls
            col_block, col_dismiss = st.columns(2)
            with col_block:
                if 2 not in st.session_state.quarantined_nodes:
                    if st.button("⛔ Quarantine Node 3", key="q_node_3"):
                        st.session_state.quarantined_nodes.add(2)
                        log_event("SOC-OPERATIONS", "Hospital 3 placed under strict quarantine. Aggregator connections revoked.", "CRITICAL")
                        st.success("✅ Node 3 isolated.")
                        st.rerun()
                else:
                    if st.button("🔓 Lift Quarantine Node 3", key="uq_node_3"):
                        st.session_state.quarantined_nodes.remove(2)
                        log_event("SOC-OPERATIONS", "Hospital 3 quarantine lifted by administrator.", "WARNING")
                        st.success("✅ Quarantine lifted.")
                        st.rerun()
            with col_dismiss:
                if st.button("🧹 Dismiss Incident Alert", key="dis_inc"):
                    # Mock dismiss
                    log_event("SOC-OPERATIONS", "Malicious update alert manually dismissed by Analyst.", "INFO")
                    st.info("Incident alert dismissed.")
        else:
            st.success("✅ **SOC Health:** No active critical incidents detected in the network.")
            
        # Quarantine Registry
        st.markdown("**Quarantine Status Registry:**")
        if st.session_state.quarantined_nodes:
            for node_idx in st.session_state.quarantined_nodes:
                st.error(f"🚫 Hospital {node_idx + 1} is currently ISOLATED from federated rounds.")
        else:
            st.caption("No nodes isolated. All clients connected.")
            
    with col_siem:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>📟 SIEM Log Stream (Real-Time Ingestion)</h4>", unsafe_allow_html=True)
        
        # Render terminal-like log viewer
        log_box_style = """
        background: #0f172a;
        color: #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.75rem;
        padding: 1rem;
        border-radius: 8px;
        height: 300px;
        overflow-y: scroll;
        border: 1px solid #1e293b;
        """
        
        log_lines = []
        for log in st.session_state.siem_logs:
            color = "#38bdf8" # Blue for info
            if log['level'] == "WARNING":
                color = "#fbbf24" # Yellow
            elif log['level'] in ["CRITICAL", "ERROR"]:
                color = "#ef4444" # Red
                
            line = f"<span style='color:#64748b;'>[{log['time']}]</span> <span style='color:#a78bfa;'>[{log['source']}]</span> <span style='color:{color}; font-weight:bold;'>[{log['level']}]</span> {log['event']}"
            log_lines.append(line)
            
        logs_html = "<br>".join(log_lines)
        st.markdown(f'<div style="{log_box_style}">{logs_html}</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Clear Event Logs", key="clear_logs"):
            st.session_state.siem_logs = [
                {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": "SIEM", "event": "Logs cleared.", "level": "INFO"}
            ]
            st.rerun()


def render_pam_section():
    """Render Privileged Access Management (PAM) simulation."""
    st.markdown('<h3 style="color:#1a365d;">🔑 PAM (Privileged Access Management)</h3>', unsafe_allow_html=True)
    st.markdown(
        "Demonstrate how access to highly-privileged client models and servers is controlled. "
        "Enforces **Just-In-Time (JIT)** elevation and audits privileged commands."
    )
    
    col_pam_info, col_jit = st.columns([1, 1.2])
    
    with col_pam_info:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>🔒 Privilege Session Status</h4>", unsafe_allow_html=True)
        
        session = st.session_state.pam_session
        if session["is_elevated"]:
            remaining = int(session["elevated_until"] - time.time())
            if remaining <= 0:
                session["is_elevated"] = False
                log_event("PAM", "Privileged JIT access expired automatically.", "INFO")
                st.rerun()
            st.success(f"🔓 **Admin Session Elevated:** Session valid for {remaining} seconds.")
            if st.button("🔒 Revoke Elevation Immediately", key="btn_revoke_pam"):
                session["is_elevated"] = False
                session["audit_trail"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Admin voluntarily revoked JIT privilege.")
                log_event("PAM", "Admin elevated access manually revoked.", "WARNING")
                st.rerun()
        else:
            st.warning("🔒 **Current Session Level:** Auditor (Read-only access to logs. Gradient updates and server configs are locked).")
            
        st.markdown("**Session Audit Trail:**")
        audit_lines = "<br>".join([f"• {x}" for x in session["audit_trail"][-6:]])
        st.markdown(f"<div style='font-size:0.8rem; background:white; padding:0.5rem; border-radius:6px; border:1px solid #edf2f7; color:#4a5568;'>{audit_lines}</div>", unsafe_allow_html=True)
        
    with col_jit:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>🛠️ Request JIT Admin Elevation</h4>", unsafe_allow_html=True)
        session = st.session_state.pam_session
        
        if not session["is_elevated"]:
            if not session["request_mfa"]:
                if st.button("🔑 Request 60-Second Admin JIT Elevation", key="req_jit"):
                    session["request_mfa"] = True
                    # Generate a mock token
                    session["mfa_token"] = str(random.randint(100000, 999999))
                    st.rerun()
            else:
                st.markdown(f"**Mock MFA Token Generated on Administrator Device:** `<strong style='color:#2b6cb0; font-size:1.2rem;'>{session['mfa_token']}</strong>`", unsafe_allow_html=True)
                mfa_input = st.text_input("Enter 6-Digit Verification Code", max_chars=6)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit Code", key="sub_mfa"):
                        if mfa_input == session["mfa_token"]:
                            session["is_elevated"] = True
                            session["request_mfa"] = False
                            session["elevated_until"] = time.time() + 60
                            session["mfa_attempts"] = 0
                            session["audit_trail"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Elevated to Administrator level via MFA verification.")
                            log_event("PAM", "JIT privilege elevation approved via Multi-Factor Verification.", "WARNING")
                            st.success("Elevation approved!")
                            st.rerun()
                        else:
                            session["mfa_attempts"] += 1
                            if session["mfa_attempts"] >= 3:
                                session["request_mfa"] = False
                                session["mfa_attempts"] = 0
                                session["audit_trail"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Elevation blocked: too many failed MFA attempts.")
                                log_event("PAM", "Brute-force security alert triggered: failed MFA attempts.", "CRITICAL")
                                st.error("MFA failed. Request canceled.")
                            else:
                                st.error(f"Invalid code. Attempt {session['mfa_attempts']}/3")
                with col2:
                    if st.button("Cancel Request", key="can_pam"):
                        session["request_mfa"] = False
                        st.rerun()
        else:
            st.info("💡 **Administrator Actions Unlocked:** you can now perform maintenance commands on client nodes:")
            cmd = st.selectbox("Select Maintenance Command", ["Reset Gradient Optimizer Weights", "Flush Client Model Cache", "Download raw hospital connection reports"])
            if st.button("Run Command", key="run_pam_cmd"):
                session["audit_trail"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Execute command: '{cmd}'")
                log_event("PAM", f"Privileged command executed: '{cmd}'", "WARNING")
                st.success(f"Success: Ran '{cmd}' on all active clusters.")


def render_compliance_section():
    """Render ITIL v4 & COBIT 2019 Compliance Mapping and PDF builder."""
    st.markdown('<h3 style="color:#1a365d;">🏛️ Enterprise Governance: ITIL & COBIT</h3>', unsafe_allow_html=True)
    st.markdown(
        "Evaluate compliance and risk controls mapped against standard frameworks. "
        "Generate a downloadable audit report for security directors."
    )
    
    col_check, col_pdf = st.columns([1.2, 1])
    
    with col_check:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>📋 COBIT 2019 Alignment Audit</h4>", unsafe_allow_html=True)
        
        # Render interactive checklist
        st.checkbox("🟢 **APO14 (Data Management):** Compliant (Clinical raw files strictly isolated in local nodes)", value=True, disabled=True)
        st.checkbox("🟢 **EDM03 (Risk Optimization):** Compliant (Validation outlier defense mechanism filters adversarial weight feeds)", value=True, disabled=True)
        
        # Zero trust policy dynamic state
        mtls_ok = st.session_state.zt_policies["mtls"]
        posture_ok = st.session_state.zt_policies["device_compliance"]
        st.checkbox(
            f"{'🟢' if mtls_ok else '❌'} **DSS05 (Security Management):** {'Compliant' if mtls_ok else 'Non-Compliant'} (Mutual TLS client verification is {'active' if mtls_ok else 'inactive'})", 
            value=mtls_ok, 
            disabled=True
        )
        st.checkbox(
            f"{'🟢' if posture_ok else '❌'} **DSS05.02 (Device Posture):** {'Compliant' if posture_ok else 'Non-Compliant'} (Device security checklist is {'active' if posture_ok else 'inactive'})", 
            value=posture_ok, 
            disabled=True
        )
        
        st.markdown("---")
        st.markdown("<h4 style='font-size:1.05rem;'>📚 ITIL v4 Practice Framework</h4>", unsafe_allow_html=True)
        st.markdown("""
        * **Service Design:** Privacy-by-design built directly into the federated weight serialization algorithms.
        * **Incident Management:** Automatic quarantine pathways trigger during anomalous gradient weight feedback cycles (SOC isolation).
        * **Change Enablement:** Controlled round iteration cycles ensure central model changes are evaluated against reference hold-out sets before production release.
        """)
        
    with col_pdf:
        st.markdown("<h4 style='font-size:1.05rem; margin-top:0;'>📥 Generate Auditor Report</h4>", unsafe_allow_html=True)
        st.write("Compile all current settings, logs, and compliance statuses into an official PDF Audit report:")
        
        # Generate the PDF
        try:
            pdf_data = generate_governance_pdf()
            
            # Layout spacing
            st.markdown("""
            <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); padding: 1.25rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(56, 161, 105, 0.3); text-align: center;">
                <p style="color: white !important; font-weight: 700; font-size: 1.1rem; margin: 0 !important;">Audit Sheet Prepared</p>
                <p style="color: rgba(255,255,255,0.85) !important; font-size: 0.85rem; margin: 0.25rem 0 0 0 !important;">FedXRay_Governance_Audit.pdf</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download Compliance Report PDF",
                data=pdf_data,
                file_name="FedXRay_Governance_Audit.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Failed to generate audit report: {e}")
