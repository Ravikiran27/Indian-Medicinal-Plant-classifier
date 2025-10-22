"""
Gemini-powered Medicinal Plant Information System
Provides medicinal uses, toxicity warnings, and regional language support
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Configure Gemini API - Hardcoded for reliability
GEMINI_API_KEY = "AIzaSyBvbX6Q5_AqXwayhkd2XW8z9aK74EcmYDo"
genai.configure(api_key=GEMINI_API_KEY)


@st.cache_data(show_spinner=False, ttl=3600)
def get_plant_information(plant_name: str, language: str = "english") -> Dict:
    """
    Get comprehensive information about a medicinal plant using Gemini AI.
    
    Args:
        plant_name: Scientific or common name of the plant
        language: Target language (english, kannada)
    
    Returns:
        Dictionary with medicinal uses, toxicity info, and translations
    """
    
    if not GEMINI_API_KEY:
        return {
            "medicinal_uses": ["API key not configured"],
            "toxicity_warning": "Configure GEMINI_API_KEY to see safety information",
            "side_effects": [],
            "kannada_name": "",
            "kannada_info": "",
            "safe_usage": "",
            "error": True
        }
    
    try:
        # Use gemini-2.5-flash (stable version available in the API)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prompt for comprehensive plant information
        prompt = f"""
You are an expert in Indian Ayurvedic medicine and medicinal plants. Provide detailed, accurate information about: {plant_name}

Please structure your response EXACTLY as follows:

**TOP 3 MEDICINAL USES:**
1. [Primary medicinal use with specific ailments]
2. [Secondary medicinal use with specific ailments]
3. [Third medicinal use with specific ailments]

**TOXICITY & SAFETY:**
Risk Level: [Low/Moderate/High]
[Detailed toxicity information, contraindications, and who should avoid it]

**COMMON SIDE EFFECTS IF MISUSED:**
- [Side effect 1]
- [Side effect 2]
- [Side effect 3]

**SAFE USAGE GUIDELINES:**
[How to use safely, recommended dosage forms, duration limits]

**KANNADA INFORMATION:**
Kannada Name: [ಕನ್ನಡ ಹೆಸರು]
Traditional Use: [Brief description in English about traditional Kannada/Karnataka usage]

**TRADITIONAL KNOWLEDGE:**
[Brief note about traditional Ayurvedic or regional significance]

Be precise, evidence-based, and include traditional knowledge. If the plant is not medicinal or information is uncertain, clearly state that.
"""
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Parse the response
        info = {
            "raw_response": response_text,
            "medicinal_uses": [],
            "toxicity_warning": "",
            "risk_level": "Unknown",
            "side_effects": [],
            "safe_usage": "",
            "kannada_name": "",
            "kannada_info": "",
            "traditional_knowledge": "",
            "error": False
        }
        
        # Extract medicinal uses
        import re
        if "**TOP 3 MEDICINAL USES:**" in response_text:
            # Split more carefully - look for the next major section header
            start_idx = response_text.find("**TOP 3 MEDICINAL USES:**") + len("**TOP 3 MEDICINAL USES:**")
            # Find the next section (TOXICITY, KANNADA, etc.)
            next_section_patterns = ["**TOXICITY", "**COMMON SIDE", "**SAFE USAGE", "**KANNADA"]
            end_idx = len(response_text)
            for pattern in next_section_patterns:
                idx = response_text.find(pattern, start_idx)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            
            uses_section = response_text[start_idx:end_idx]
            uses = []
            for line in uses_section.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Check if line starts with a number (e.g., "1.", "1 ", "1)") or bullet point
                if line and len(line) > 2:  # Must have content after the number
                    first_char = line[0]
                    if first_char.isdigit() or first_char in ['-', '•', '*']:
                        # Remove the number prefix but keep the rest
                        cleaned = re.sub(r'^[\d\.\)\s]+', '', line, count=1).strip()
                        # Remove any ** markdown
                        cleaned = cleaned.replace('**', '')
                        if cleaned and len(cleaned) > 3:  # Must have meaningful content
                            uses.append(cleaned)
            info["medicinal_uses"] = uses[:3] if uses else ["Information not available"]
        
        # Extract toxicity info
        if "**TOXICITY & SAFETY:**" in response_text:
            toxicity_section = response_text.split("**TOXICITY & SAFETY:**")[1].split("**")[0]
            info["toxicity_warning"] = toxicity_section.strip()
            
            if "Risk Level:" in toxicity_section:
                risk_line = [l for l in toxicity_section.split("\n") if "Risk Level:" in l]
                if risk_line:
                    info["risk_level"] = risk_line[0].split("Risk Level:")[1].strip()
        
        # Extract side effects
        if "**COMMON SIDE EFFECTS IF MISUSED:**" in response_text:
            effects_section = response_text.split("**COMMON SIDE EFFECTS IF MISUSED:**")[1].split("**")[0]
            effects = [line.strip().lstrip("- ").strip() for line in effects_section.split("\n") if line.strip().startswith("-")]
            info["side_effects"] = effects if effects else ["No major side effects documented"]
        
        # Extract safe usage
        if "**SAFE USAGE GUIDELINES:**" in response_text:
            usage_section = response_text.split("**SAFE USAGE GUIDELINES:**")[1].split("**")[0]
            info["safe_usage"] = usage_section.strip()
        
        # Extract Kannada information
        if "**KANNADA INFORMATION:**" in response_text:
            kannada_section = response_text.split("**KANNADA INFORMATION:**")[1].split("**")[0]
            for line in kannada_section.split("\n"):
                if "Kannada Name:" in line:
                    info["kannada_name"] = line.split("Kannada Name:")[1].strip()
                elif "Traditional Use:" in line:
                    info["kannada_info"] = line.split("Traditional Use:")[1].strip()
        
        # Extract traditional knowledge
        if "**TRADITIONAL KNOWLEDGE:**" in response_text:
            trad_section = response_text.split("**TRADITIONAL KNOWLEDGE:**")[1].strip()
            # Take until next section or end
            info["traditional_knowledge"] = trad_section.split("**")[0].strip() if "**" in trad_section else trad_section.strip()
        
        return info
        
    except Exception as e:
        st.error(f"Error fetching plant information: {str(e)}")
        return {
            "medicinal_uses": ["Error fetching information"],
            "toxicity_warning": f"Could not retrieve safety information: {str(e)}",
            "side_effects": [],
            "kannada_name": "",
            "kannada_info": "",
            "safe_usage": "",
            "traditional_knowledge": "",
            "error": True
        }


def display_plant_info(plant_name: str, confidence: float):
    """Display comprehensive plant information in Streamlit UI"""
    
    st.markdown("---")
    st.subheader(f"ℹ️ About {plant_name}")
    
    with st.spinner("🔍 Fetching medicinal information from Gemini AI..."):
        info = get_plant_information(plant_name)
    
    if info.get("error"):
        st.warning("⚠️ Could not fetch complete information. Please check your API configuration.")
        return
    
    # Create tabs for organized information
    tab1, tab2, tab3, tab4 = st.tabs(["🌿 Medicinal Uses", "⚠️ Safety & Toxicity", "🗣️ ಕನ್ನಡ (Kannada)", "📚 Traditional Knowledge"])
    
    with tab1:
        st.markdown("### Top 3 Medicinal Uses")
        if info["medicinal_uses"]:
            for i, use in enumerate(info["medicinal_uses"], 1):
                st.markdown(f"**{i}.** {use}")
        else:
            st.info("Medicinal use information not available.")
        
        if info.get("safe_usage"):
            st.markdown("### ✅ Safe Usage Guidelines")
            st.info(info["safe_usage"])
    
    with tab2:
        # Risk level indicator
        risk_level = info.get("risk_level", "Unknown")
        if "Low" in risk_level:
            st.success(f"🟢 **Risk Level:** {risk_level}")
        elif "Moderate" in risk_level:
            st.warning(f"🟡 **Risk Level:** {risk_level}")
        elif "High" in risk_level:
            st.error(f"🔴 **Risk Level:** {risk_level}")
        else:
            st.info(f"**Risk Level:** {risk_level}")
        
        st.markdown("### ⚠️ Toxicity & Contraindications")
        st.warning(info.get("toxicity_warning", "Safety information not available."))
        
        if info.get("side_effects"):
            st.markdown("### 🔸 Common Side Effects if Misused")
            for effect in info["side_effects"]:
                st.markdown(f"- {effect}")
        
        st.markdown("""
        ---
        **⚕️ Medical Disclaimer:** This information is for educational purposes only. 
        Always consult with a qualified healthcare provider or Ayurvedic practitioner before using any medicinal plant.
        """)
    
    with tab3:
        if info.get("kannada_name"):
            st.markdown(f"### ಹೆಸರು (Name)")
            st.markdown(f"## {info['kannada_name']}")
        
        if info.get("kannada_info"):
            st.markdown(f"### ಸಾಂಪ್ರದಾಯಿಕ ಬಳಕೆ (Traditional Use)")
            st.info(info['kannada_info'])
        
        if not info.get("kannada_name") and not info.get("kannada_info"):
            st.info("ಕನ್ನಡ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ (Kannada information not available)")
        
        st.markdown("---")
        st.caption("💡 Regional language support helps preserve traditional knowledge and makes medicinal plant information accessible to local communities.")
    
    with tab4:
        if info.get("traditional_knowledge"):
            st.markdown(info["traditional_knowledge"])
        else:
            st.info("Traditional knowledge information not available.")
        
        st.markdown("---")
        st.markdown("""
        ### 🌍 Preserving Traditional Knowledge Through AI
        
        This system combines:
        - **Ancient Ayurvedic Wisdom**: Thousands of years of traditional Indian medicine
        - **Modern AI Technology**: Google Gemini for accurate, contextual information
        - **Community Accessibility**: Regional language support for local communities
        
        By documenting and sharing this knowledge, we help preserve traditional medicinal practices 
        while promoting safe and informed usage.
        """)


def get_safety_color(risk_level: str) -> str:
    """Return color code based on risk level"""
    if "Low" in risk_level:
        return "#28a745"  # Green
    elif "Moderate" in risk_level:
        return "#ffc107"  # Yellow
    elif "High" in risk_level:
        return "#dc3545"  # Red
    else:
        return "#6c757d"  # Gray
