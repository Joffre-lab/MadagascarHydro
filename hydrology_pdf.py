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
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#0284C7'),
        spaceBefore=6,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    intro_style = ParagraphStyle(
        'IntroText',
        parent=body_style,
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor('#475569'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        fontSize=7.8,
        leftIndent=8,
        spaceAfter=2
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=body_style,
        fontSize=6.8,
        leading=9,
        textColor=colors.HexColor('#64748B')
    )

    elements = []

    
    elements.append(Paragraph("🇲🇬 Rapport d'Étude Hydrologique & Modélisation IA", title_style))
    elements.append(Paragraph(f"<b>Exutoire :</b> Lat {center_coords[0]:.5f}°, Lon {center_coords[1]:.5f}° | <b>Projection :</b> EPSG:{metrics.get('target_epsg', '32738')}", body_style))
    elements.append(Spacer(1, 6))

   
    elements.append(Paragraph("1. Caractéristiques Morphologiques & Occupation du Sol", h2_style))
    elements.append(Paragraph("Les paramètres géomorphométriques du bassin versant sont dérivés du Modèle Numérique de Terrain (FABDEM 30m). Ils caractérisent la réponse physique du bassin à l'exutoire.", intro_style))
    

    raw_tc = (
        metrics.get('tc_giandotti_hrs') or 
        metrics.get('tc_giandotti') or 
        metrics.get('tc_hours') or 
        metrics.get('tc_passini_hrs') or 
        metrics.get('tc') or 0
    )

    try:
        tc_val = float(raw_tc)
    except (ValueError, TypeError):
        tc_val = 0.0

    if tc_val <= 0:
        _area = float(metrics.get('area_km2', 0))
        _l_rect = float(metrics.get('l_rect_km', 0))
        _z_moy = float(metrics.get('z_moy', metrics.get('mean_elev', 0)))
        _z_min = float(metrics.get('min_elev', 0))
        _dz = max(_z_moy - _z_min, 1.0)
        if _area > 0 and _l_rect > 0:
            tc_val = (4.0 * (_area ** 0.5) + 1.5 * _l_rect) / (0.8 * (_dz ** 0.5))
            tc_val = max(round(tc_val, 2), 0.10)

    z_moy_val = metrics.get('z_moy', metrics.get('mean_elev', 0))
    dd_val = metrics.get('drainage_density', metrics.get('dd', 0))

    geom_data = [
        ["Caractéristique Morphologique", "Valeur"],
        ["Superficie du Bassin (S)", f"{metrics.get('area_km2', 0):,.2f} km²"],
        ["Périmètre (P)", f"{metrics.get('perimeter_km', 0):,.2f} km"],
        ["Altitude Minimale / Maximale", f"{metrics.get('min_elev', 0):.0f} m / {metrics.get('max_elev', 0):.0f} m"],
        ["Altitude Moyenne (Z_moy)", f"{z_moy_val:.0f} m"],
        ["Pente Globale (Ig)", f"{metrics.get('slope_m_km', 0):.2f} m/km"],
        ["Longueur du rectangle équivalent (Leq)", f"{metrics.get('l_rect_km', 0):,.2f} km"],
        ["Indice de Compacité de Gravelius (Kc)", f"{metrics.get('kc', 0):.2f}"],
        ["Temps de Concentration (Giandotti Tc)", f"{tc_val:.2f} h"],
        ["Densité de Drainage (Dd)", f"{dd_val:.2f} km/km²"]
    ]
    t_geom = Table(geom_data, colWidths=[11*cm, 7.6*cm])
    t_geom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(t_geom)
    elements.append(Spacer(1, 4))

    # Tableau Occupation du Sol
    lc_df = metrics.get('lc_df')
    if isinstance(lc_df, pd.DataFrame) and not lc_df.empty:
        elements.append(Paragraph("Répartition des classes d'occupation du sol issues du produit haute résolution ESA WorldCover (10m) :", intro_style))
        lc_table_data = [["Type d'Occupation du Sol (ESA WorldCover)", "Surface (km²)", "Proportion (%)"]]
        for _, row in lc_df.iterrows():
            lc_table_data.append([
                str(row["Type de sol"]),
                f"{row['Surface (km²)']:.2f}",
                f"{row['Proportion (%)']:.1f} %"
            ])
        t_lc = Table(lc_table_data, colWidths=[9.6*cm, 4.5*cm, 4.5*cm])
        t_lc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ]))
        elements.append(t_lc)
    
    elements.append(Spacer(1, 6))

    # --- 2. PLUVIOMÉTRIE DE PROJET ---
    elements.append(Paragraph("2. Pluviométrie de Projet", h2_style))
    elements.append(Paragraph(
        f"La pluviométrie régionale s'appuie sur la grille pluviométrique spatiale CHIRPS v2.0 calibré avec 21 stations Pluviométriques implantés à Madagascar. "
        f"Les valeurs (moyenne annuelle de <b>{metrics.get('p_annuelle_mm', 0):,.0f} mm/an</b>, distributions mensuelles et maxima 24h) "
        f"sont calculées par surcouche spatiale (spatial masking) en effectuant la moyenne exacte de tous les pixels raster interceptés par le polygone du bassin versant.", 
        intro_style
    ))
    
    if 'p_mensuelles' in metrics and metrics['p_mensuelles']:
        m_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
        m_vals = [f"{v:.1f}" for v in metrics['p_mensuelles']]
        monthly_table_data = [m_labels[:6], m_vals[:6], m_labels[6:], m_vals[6:]]
        t_month = Table(monthly_table_data, colWidths=[3.1*cm]*6)
        t_month.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E0F2FE')),
            ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#E0F2FE')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,2), (-1,2), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ]))
        elements.append(t_month)
        elements.append(Spacer(1, 4))

    elements.append(Paragraph("Hauteurs d'eau maximales en 24h estimées pour différentes périodes de retour (Loi statistique de Gumbel) :", intro_style))
    p_design = metrics.get('p_design', {10: 0, 25: 0, 50: 0, 100: 0})
    p_data = [
        ["Période de Retour (T)", "10 ans", "25 ans", "50 ans", "100 ans"],
        ["Pluie de Projet P24h (mm)", 
         f"{p_design.get(10, 0):.1f}", f"{p_design.get(25, 0):.1f}", 
         f"{p_design.get(50, 0):.1f}", f"{p_design.get(100, 0):.1f}"]
    ]
    t_p = Table(p_data, colWidths=[6.6*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    t_p.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#FEF3C7')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
    ]))
    elements.append(t_p)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("3. Simulation des Débits de Crue par Intelligence Artificielle", h2_style))
    
    in_domain = metrics.get('in_domain', False)
    q_ml = metrics.get('q_dict')
    versant = metrics.get('versant', 'Inconnu')

    if in_domain and q_ml is not None:
        elements.append(Paragraph(f"Estimation des débits de pointe par l'algorithme <b>Gradient Boosting Regressor</b> (Région : <i>{versant}</i>) :", intro_style))
        q_ml_data = [
            ["Période de Retour (T)", "Q10", "Q25", "Q50", "Q100"],
            ["Débit de Pointe IA (m³/s)", 
             f"{q_ml.get(10, 0):,.1f}", f"{q_ml.get(25, 0):,.1f}", 
             f"{q_ml.get(50, 0):,.1f}", f"{q_ml.get(100, 0):,.1f}"]
        ]
        t_q_ml = Table(q_ml_data, colWidths=[6.6*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        t_q_ml.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DCFCE7')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ]))
        elements.append(t_q_ml)
        elements.append(Spacer(1, 4))

       
        egv = metrics.get('egv_params', {})
        elements.append(Paragraph("Variables explicatives physiques et hydro-climatiques injectées dans le modèle :", intro_style))
        egv_data = [
            ["Indice Exondation (E)", "Indice Imperméab. (G)", "Indice Végétation (V)", "Indice Pente (Ig)", "Indice Pluie (P10)", "Surface (S)"],
            [f"{egv.get('E', '—')}", f"{egv.get('G', '—')}", f"{egv.get('V', '—')}", 
             f"{egv.get('Ig_m_km', '—')} m/km", f"{egv.get('P10_mm', '—')} mm", f"{egv.get('Surface_km2', '—')} km²"]
        ]
        t_egv = Table(egv_data, colWidths=[3.1*cm]*6)
        t_egv.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ]))
        elements.append(t_egv)
    else:
        elements.append(Paragraph(f"<b>Avertissement :</b> Le bassin (Région : <i>{versant}</i>) dépasse les bornes de validité du modèle IA entraîné.", body_style))

    elements.append(Spacer(1, 6))

    
    elements.append(Paragraph("4. Sources des Données & Méthodologie", h2_style))
    elements.append(Paragraph("• <b>MNT & Altimétrie :</b> Copernicus / FABDEM 30m (correction haute précision de la canopée et du bâti).", bullet_style))
    elements.append(Paragraph("• <b>Occupation du Sol :</b> ESA WorldCover v200 (10m), extraction vectorielle pixel à pixel.", bullet_style))
    elements.append(Paragraph("• <b>Pluviométrie :</b> CHIRPS v2.0 (Climate Hazards Center) calculée par moyenne spatiale exacte sur la grille raster interceptée.", bullet_style))
    elements.append(Paragraph("• <b>Modélisation IA :</b> Algorithme Gradient Boosting Regressor entraîné sur 96 bassins versants jaugés à Madagascar (LOO KGE = 0.915).", bullet_style))

    elements.append(Spacer(1, 8))

    
    disclaimer_html = (
        "<b>⚠️ Avertissement & Conditions d'Utilisation :</b><br/>"
        "Ce rapport est généré automatiquement par la plateforme MadaHydro à des fins d'aide à la décision et d'études préliminaires d'ingénierie. "
        "Bien que les modèles numériques (MNT FABDEM 30m, IA Gradient Boosting, formule de Giandotti) s'appuient sur des bases géospatiales validées, "
        "ces résultats d'évaluation rapide ne remplacent pas une étude d'exécution détaillée, une campagne de jaugeage in-situ ni des vérifications terrain. "
        "L'utilisateur conserve l'entière responsabilité du contrôle et de l'interprétation de ces données pour le dimensionnement d'ouvrages hydrauliques ou la gestion des risques d'inondation."
    )
    t_disc = Table([[Paragraph(disclaimer_html, disclaimer_style)]], colWidths=[18.6*cm])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_disc)

    doc.build(elements)
    pdf_val = buffer.getvalue()
    buffer.close()
    return pdf_val