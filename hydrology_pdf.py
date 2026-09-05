import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf_report(metrics, center_coords):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=8
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0284C7'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=10,
        spaceAfter=3
    )

    elements = []

    # --- EN-TÊTE DU RAPPORT ---
    elements.append(Paragraph("🇲🇬 Rapport d'Étude Hydrologique & Modélisation IA", title_style))
    elements.append(Paragraph(f"Exutoire : Lat {center_coords[0]:.4f}°, Lon {center_coords[1]:.4f}° | Projection EPSG:{metrics.get('target_epsg', '32738')}", body_style))
    elements.append(Spacer(1, 8))

    # --- 1. MORPHOLOGIE & OCCUPATION DU SOL ---
    elements.append(Paragraph("1. Caractéristiques du Bassin Versant & Occupation du Sol", h2_style))
    
    geom_data = [
        ["Caractéristique Morphologique", "Valeur"],
        ["Superficie du Bassin (S)", f"{metrics.get('area_km2', 0):,.2f} km²"],
        ["Périmètre (P)", f"{metrics.get('perimeter_km', 0):,.2f} km"],
        ["Altitude Minimale", f"{metrics.get('min_elev', 0):.0f} m"],
        ["Altitude Maximale", f"{metrics.get('max_elev', 0):.0f} m"],
        ["Pente Globale (Ig)", f"{metrics.get('slope_m_km', 0):.2f} m/km"],
        ["Longueur du rectangle équivalent (Leq)", f"{metrics.get('l_rect_km', 0):,.2f} km"],
        ["Indice de Compacité de Gravelius (Kc)", f"{metrics.get('kc', 0):.2f}"],
        ["Temps de Concentration (Passini Tc)", f"{metrics.get('tc_passini_hrs', 0):.2f} h"]
    ]
    t_geom = Table(geom_data, colWidths=[10*cm, 8*cm])
    t_geom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(t_geom)
    elements.append(Spacer(1, 6))

    # Tableau Occupation du Sol (ESA WorldCover)
    lc_df = metrics.get('lc_df')
    if isinstance(lc_df, pd.DataFrame) and not lc_df.empty:
        lc_table_data = [["Type d'Occupation du Sol (ESA WorldCover)", "Surface (km²)", "Proportion (%)"]]
        for _, row in lc_df.iterrows():
            lc_table_data.append([
                str(row["Type de sol"]),
                f"{row['Surface (km²)']:.2f}",
                f"{row['Proportion (%)']:.1f} %"
            ])
        t_lc = Table(lc_table_data, colWidths=[9*cm, 4.5*cm, 4.5*cm])
        t_lc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        elements.append(t_lc)
    
    elements.append(Spacer(1, 8))

    # --- 2. PLUVIOMÉTRIE DE PROJET ---
    elements.append(Paragraph("2. Pluviométrie de Projet (CHIRPS & Loi de Gumbel)", h2_style))
    elements.append(Paragraph(f"Pluie Moyenne Annuelle Totale (CHIRPS) : <b>{metrics.get('p_annuelle_mm', 0):,.0f} mm/an</b>", body_style))
    elements.append(Spacer(1, 4))
    
    if 'p_mensuelles' in metrics and metrics['p_mensuelles']:
        m_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
        m_vals = [f"{v:.1f}" for v in metrics['p_mensuelles']]
        monthly_table_data = [m_labels[:6], m_vals[:6], m_labels[6:], m_vals[6:]]
        t_month = Table(monthly_table_data, colWidths=[3*cm]*6)
        t_month.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E0F2FE')),
            ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#E0F2FE')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,2), (-1,2), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        elements.append(t_month)
        elements.append(Spacer(1, 4))

    p_design = metrics.get('p_design', {10: 0, 25: 0, 50: 0, 100: 0})
    p_data = [
        ["Période de Retour (T)", "10 ans", "25 ans", "50 ans", "100 ans"],
        ["Pluie de Projet P24h (mm)", 
         f"{p_design.get(10, 0):.1f}", f"{p_design.get(25, 0):.1f}", 
         f"{p_design.get(50, 0):.1f}", f"{p_design.get(100, 0):.1f}"]
    ]
    t_p = Table(p_data, colWidths=[6*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    t_p.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#FEF3C7')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_p)
    elements.append(Spacer(1, 8))

    # --- 3. DÉBITS DE CRUE & IA ---
    elements.append(Paragraph("3. Simulation des Débits de Crue (IA & Modèles Classiques)", h2_style))
    
    in_domain = metrics.get('in_domain', False)
    q_ml = metrics.get('q_dict')
    versant = metrics.get('versant', 'Inconnu')

    # Débits IA ML
    if in_domain and q_ml is not None:
        elements.append(Paragraph(f"<b>Modèle Machine Learning Actif</b> — Versant : {versant} (Gradient Boosting Regressor, KGE = 0.915)", body_style))
        elements.append(Spacer(1, 2))
        q_ml_data = [
            ["Période de Retour (T)", "Q10", "Q25", "Q50", "Q100"],
            ["Débit de Pointe IA (m³/s)", 
             f"{q_ml.get(10, 0):,.1f}", f"{q_ml.get(25, 0):,.1f}", 
             f"{q_ml.get(50, 0):,.1f}", f"{q_ml.get(100, 0):,.1f}"]
        ]
        t_q_ml = Table(q_ml_data, colWidths=[6*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        t_q_ml.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DCFCE7')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        elements.append(t_q_ml)
    else:
        elements.append(Paragraph(f"<b>Avertissement :</b> Le versant <i>{versant}</i> est hors du domaine d'applicabilité IA. Seules les formules classiques sont appliquées.", body_style))
    
    elements.append(Spacer(1, 4))

    # Débits Empiriques Classiques
    q_orstom = metrics.get('q_dict_orstom', {})
    orstom_method = metrics.get('orstom_method', 'ORSTOM / SOMEAH')
    elements.append(Paragraph(f"<b>Modèle Empirique Classique</b> ({orstom_method}) :", body_style))
    elements.append(Spacer(1, 2))
    q_emp_data = [
        ["Formule Empirique", "Q10 Classique", "Q100 Classique"],
        [orstom_method, f"{q_orstom.get(10, 0):,.1f} m³/s", f"{q_orstom.get(100, 0):,.1f} m³/s"]
    ]
    t_q_emp = Table(q_emp_data, colWidths=[8*cm, 5*cm, 5*cm])
    t_q_emp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_q_emp)
    elements.append(Spacer(1, 4))

    # Paramètres ML calculés
    egv = metrics.get('egv_params', {})
    elements.append(Paragraph("<b>Paramètres ML Calculés (Variables explicatives) :</b>", body_style))
    elements.append(Spacer(1, 2))
    egv_data = [
        ["Exondation (E)", "Imperméab. (G)", "Végétation (V)", "Pente (Ig)", "Pluie (P10)", "Surface (S)"],
        [f"{egv.get('E', '—')}", f"{egv.get('G', '—')}", f"{egv.get('V', '—')}", 
         f"{egv.get('Ig_m_km', '—')} m/km", f"{egv.get('P10_mm', '—')} mm", f"{egv.get('Surface_km2', '—')} km²"]
    ]
    t_egv = Table(egv_data, colWidths=[3*cm]*6)
    t_egv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_egv)
    elements.append(Spacer(1, 8))

    # --- 4. SOURCES DES DONNÉES ---
    elements.append(Paragraph("4. Sources des Données & Modélisation", h2_style))
    elements.append(Paragraph("• <b>MNT & Altimétrie :</b> Modèle Numérique de Terrain de haute précision (Copernicus / FABDEM 30m).", bullet_style))
    elements.append(Paragraph("• <b>Occupation du Sol :</b> ESA WorldCover v200 (10m), extraction spatiale pixel à pixel.", bullet_style))
    elements.append(Paragraph("• <b>Pluviométrie :</b> Série temporelle climatique CHIRPS v2.0 (Climate Hazards Center, 1981-2023).", bullet_style))
    elements.append(Paragraph("• <b>Intelligence Artificielle :</b> Algorithme Gradient Boosting Regressor (Calibré sur 96 bassins versants jaugés à Madagascar, Validation Croisée LOO KGE = 0.915).", bullet_style))

    doc.build(elements)
    pdf_val = buffer.getvalue()
    buffer.close()
    return pdf_val