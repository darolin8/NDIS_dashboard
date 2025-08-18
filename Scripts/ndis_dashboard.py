      risk_data.append({
                'location': location,
                'total_incidents': total_incidents,
                'avg_severity': avg_severity,
                'risk_score': total_incidents * avg_severity
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        fig_risk = px.scatter(
            risk_df, 
            x='total_incidents', 
            y='avg_severity',
            size='risk_score',
            color='risk_score',
            hover_name='location',
            title="Risk Matrix: Volume vs Severity by Location",
            labels={'total_incidents': 'Incident Volume', 'avg_severity': 'Average Severity Score'},
            color_continuous_scale="Reds"
        )
        
        # Add quadrant lines
        median_volume = risk_df['total_incidents'].median()
        median_severity = risk_df['avg_severity'].median()
        
        fig_risk.add_hline(y=median_severity, line_dash="dash", line_color="gray", opacity=0.5)
        fig_risk.add_vline(x=median_volume, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Age vs Incident Type Risk Analysis
        age_incident_matrix = pd.crosstab(df_filtered['age_group'], df_filtered['incident_type'])
        age_incident_pct = age_incident_matrix.div(age_incident_matrix.sum(axis=1), axis=0) * 100
        
        fig_age_risk = px.imshow(
            age_incident_pct,
            title="Incident Type Risk by Age Group (%)",
            labels=dict(x="Incident Type", y="Age Group", color="Percentage"),
            color_continuous_scale="YlOrRd"
        )
        fig_age_risk.update_xaxes(tickangle=45)
        st.plotly_chart(fig_age_risk, use_container_width=True)
    
    # Risk Factors Analysis
    st.subheader("📊 Risk Factor Analysis")
    
    risk_factors = {}
    
    # Weekend vs weekday risk
    if len(df_filtered[df_filtered['is_weekend']]) > 0:
        risk_factors['Weekend Incidents'] = df_filtered[df_filtered['is_weekend']]['severity_score'].mean()
    
    # Night hours risk
    night_hours = list(range(22, 24)) + list(range(0, 7))
    night_incidents = df_filtered[df_filtered['hour'].isin(night_hours)]
    if len(night_incidents) > 0:
        risk_factors['Night Hours (22-06)'] = night_incidents['severity_score'].mean()
    
    # Repeat participants risk
    repeat_participants = df_filtered['participant_name'].value_counts()
    repeat_names = repeat_participants[repeat_participants > 1].index
    if len(repeat_names) > 0:
        risk_factors['Repeat Participants'] = df_filtered[df_filtered['participant_name'].isin(repeat_names)]['severity_score'].mean()
    
    # Delayed reporting risk
    if 'notification_delay' in df_filtered.columns:
        delayed_reports = df_filtered[df_filtered['notification_delay'] > 1]
        if len(delayed_reports) > 0:
            risk_factors['Delayed Reporting'] = delayed_reports['severity_score'].mean()
    
    if risk_factors:
        risk_factor_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Risk Score'])
        risk_factor_df = risk_factor_df.sort_values('Risk Score', ascending=True)
        
        fig_factors = px.bar(
            risk_factor_df,
            x='Risk Score',
            y='Factor',
            orientation='h',
            title="Risk Factor Impact on Incident Severity",
            color='Risk Score',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_factors, use_container_width=True)

elif analysis_mode == "Correlation Explorer":
    st.subheader("🔗 Interactive Correlation Analysis")
    
    # Correlation matrix heatmap
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Key Variables",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Key Correlations")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.1:  # Only show meaningful correlations
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if corr_pairs:
            corr_pairs_df = pd.DataFrame(corr_pairs)
            corr_pairs_df = corr_pairs_df.reindex(corr_pairs_df.correlation.abs().sort_values(ascending=False).index)
            
            for _, row in corr_pairs_df.head(5).iterrows():
                strength = "Strong" if abs(row['correlation']) > 0.5 else "Moderate" if abs(row['correlation']) > 0.3 else "Weak"
                direction = "Positive" if row['correlation'] > 0 else "Negative"
                
                st.markdown(f"""
                <div class="correlation-card">
                    <strong>{row['var1']} ↔ {row['var2']}</strong><br>
                    {direction} {strength}<br>
                    r = {row['correlation']:.3f}
                </div>
                """, unsafe_allow_html=True)
    
    # Interactive scatter plots
    st.subheader("🔍 Relationship Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X-axis Variable", corr_matrix.columns, index=0)
    with col2:
        y_var = st.selectbox("Y-axis Variable", corr_matrix.columns, index=1)
    with col3:
        color_var = st.selectbox("Color by", ['severity', 'location', 'incident_type', 'age_group'])
    
    if x_var != y_var:
        fig_scatter = px.scatter(
            numeric_df,
            x=x_var,
            y=y_var,
            color=color_var,
            title=f"Relationship between {x_var} and {y_var}",
            trendline="ols" if st.checkbox("Show trend line") else None,
            hover_data=['participant_name', 'incident_date']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Statistical significance test
        valid_data = numeric_df[[x_var, y_var]].dropna()
        if len(valid_data) > 2:
            correlation, p_value = stats.pearsonr(valid_data[x_var], valid_data[y_var])
            significance = "Significant" if p_value < 0.05 else "Not significant"
            
            st.info(f"**Statistical Analysis:** Correlation = {correlation:.3f}, p-value = {p_value:.3f} ({significance})")

elif analysis_mode == "Predictive Insights":
    st.subheader("🔮 Predictive Analytics & Forecasting")
    
    # Time series analysis
    monthly_incidents = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
    monthly_incidents.index = monthly_incidents.index.to_timestamp()
    
    if len(monthly_incidents) > 3:
        # Simple trend analysis
        trend_data = pd.DataFrame({
            'month': range(len(monthly_incidents)),
            'incidents': monthly_incidents.values
        })
        
        # Calculate linear trend
        z = np.polyfit(trend_data['month'], trend_data['incidents'], 1)
        trend_line = np.poly1d(z)
        
        # Create forecast
        future_months = range(len(monthly_incidents), len(monthly_incidents) + 6)
        forecast = [trend_line(m) for m in future_months]
        
        # Plot
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=monthly_incidents.index,
            y=monthly_incidents.values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Trend line
        fig_forecast.add_trace(go.Scatter(
            x=monthly_incidents.index,
            y=[trend_line(i) for i in range(len(monthly_incidents))],
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        # Forecast
        future_dates = pd.date_range(monthly_incidents.index[-1], periods=7, freq='M')[1:]
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange')
        ))
        
        fig_forecast.update_layout(
            title="Incident Trend Forecast (6 Months Ahead)",
            xaxis_title="Date",
            yaxis_title="Number of Incidents"
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Risk Prediction Calculator
    st.subheader("⚠️ Risk Prediction Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Scenario Planning")
        
        # Interactive risk calculator
        selected_age = st.slider("Participant Age", 18, 85, 35)
        selected_location = st.selectbox("Location", df['location'].unique())
        selected_time = st.selectbox("Time of Day", ["Morning (6-12)", "Afternoon (12-18)", "Evening (18-22)", "Night (22-6)"])
        is_weekend_scenario = st.checkbox("Weekend?")
        
        # Calculate risk based on historical data
        time_mapping = {
            "Morning (6-12)": list(range(6, 12)), 
            "Afternoon (12-18)": list(range(12, 18)), 
            "Evening (18-22)": list(range(18, 22)), 
            "Night (22-6)": list(range(22, 24)) + list(range(0, 6))
        }
        
        scenario_filter = (
            (df['age'] >= selected_age - 5) & (df['age'] <= selected_age + 5) &
            (df['location'] == selected_location) &
            (df['hour'].isin(time_mapping[selected_time])) &
            (df['is_weekend'] == is_weekend_scenario)
        )
        
        scenario_incidents = df[scenario_filter]
        
        if len(scenario_incidents) > 0:
            avg_severity = scenario_incidents['severity_score'].mean()
            incident_probability = len(scenario_incidents) / len(df) * 100
            
            risk_level = "Low" if avg_severity < 1.5 else "Medium" if avg_severity < 2.5 else "High" if avg_severity < 3.5 else "Critical"
            risk_color = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high", "Critical": "risk-critical"}[risk_level]
            
            st.markdown(f"""
            <div class="risk-card {risk_color}">
                <h3>Risk Assessment</h3>
                <p><strong>Risk Level: {risk_level}</strong></p>
                <p>Average Severity: {avg_severity:.2f}/4.0</p>
                <p>Historical Probability: {incident_probability:.2f}%</p>
                <p>Sample Size: {len(scenario_incidents)} incidents</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No historical data available for this scenario combination.")
    
    with col2:
        st.markdown("### 📈 Risk Trends")
        
        # Risk trend over time
        risk_over_time = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M'))['severity_score'].mean()
        risk_over_time.index = risk_over_time.index.astype(str)
        
        fig_risk_trend = px.line(
            x=risk_over_time.index,
            y=risk_over_time.values,
            title="Average Risk Score Over Time",
            labels={'x': 'Month', 'y': 'Average Risk Score'}
        )
        fig_risk_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_risk_trend, use_container_width=True)

elif analysis_mode == "Performance Analytics":
    st.subheader("📊 Performance Analytics Dashboard")
    
    # Check which columns are available
    has_reporter_type = 'reporter_type' in df_filtered.columns
    has_medical_attention = 'medical_attention' in df_filtered.columns
    
    if has_reporter_type:
        # Reporter Performance Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 Reporter Performance")
            
            reporter_stats = df_filtered.groupby('reporter_type').agg({
                'incident_id': 'count',
                'notification_delay': 'mean',
                'severity_score': 'mean'
            }).round(2)
            
            reporter_stats.columns = ['Total Reports', 'Avg Delay (days)', 'Avg Severity']
            
            fig_reporter = px.scatter(
                reporter_stats.reset_index(),
                x='Avg Delay (days)',
                y='Avg Severity',
                size='Total Reports',
                color='Total Reports',
                hover_name='reporter_type',
                title="Reporter Performance: Speed vs Quality",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_reporter, use_container_width=True)
        
        with col2:
            st.markdown("### 🏢 Location Performance")
            
            # Calculate medical rate only if column exists
            agg_dict = {
                'incident_id': 'count',
                'notification_delay': 'mean',
                'severity_score': 'mean'
            }
            
            if has_medical_attention:
                agg_dict['medical_attention'] = lambda x: (x == 'Yes').mean()
                
            location_stats = df_filtered.groupby('location').agg(agg_dict).round(2)
            
            if has_medical_attention:
                location_stats.columns = ['Total Incidents', 'Avg Delay', 'Avg Severity', 'Medical Rate']
                # Performance score (lower is better)
                location_stats['Performance Score'] = (
                    location_stats['Avg Delay'] * 0.3 + 
                    location_stats['Avg Severity'] * 0.4 + 
                    location_stats['Medical Rate'] * 0.3
                )
            else:
                location_stats.columns = ['Total Incidents', 'Avg Delay', 'Avg Severity']
                # Performance score without medical rate
                location_stats['Performance Score'] = (
                    location_stats['Avg Delay'] * 0.5 + 
                    location_stats['Avg Severity'] * 0.5
                )
            
            fig_location = px.bar(
                location_stats.reset_index().sort_values('Performance Score'),
                x='location',
                y='Performance Score',
                title="Location Performance Score (Lower = Better)",
                color='Performance Score',
                color_continuous_scale="RdYlGn_r"
            )
            fig_location.update_xaxes(tickangle=45)
            st.plotly_chart(fig_location, use_container_width=True)
    else:
        st.info("👥 Reporter performance analysis requires 'reporter_type' column in your data")
        
        # Show basic location analysis instead
        st.markdown("### 🏢 Location Analysis")
        location_stats = df_filtered.groupby('location').agg({
            'incident_id': 'count',
            'severity_score': 'mean'
        }).round(2)
        location_stats.columns = ['Total Incidents', 'Avg Severity']
        
        fig_location = px.bar(
            location_stats.reset_index(),
            x='location',
            y='Total Incidents',
            color='Avg Severity',
            title="Incidents by Location",
            color_continuous_scale="Reds"
        )
        fig_location.update_xaxes(tickangle=45)
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Compliance Dashboard
    st.subheader("✅ Compliance Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Notification compliance
        if 'notification_delay' in df_filtered.columns:
            compliance_by_severity = df_filtered.groupby('severity').apply(
                lambda x: (x['notification_delay'] <= 1).mean() * 100
            )
            
            fig_compliance = px.bar(
                x=compliance_by_severity.index,
                y=compliance_by_severity.values,
                title="Notification Compliance by Severity (%)",
                color=compliance_by_severity.values,
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_compliance, use_container_width=True)
        else:
            st.info("📅 Notification compliance requires 'notification_delay' column")
    
    with col2:
        if has_medical_attention:
            # Medical attention compliance
            medical_compliance = df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]
            medical_rate = (medical_compliance['medical_attention'] == 'Yes').mean() * 100
            
            fig_medical = go.Figure(go.Indicator(
                mode="gauge+number",
                value=medical_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Medical Attention Rate for High/Critical Incidents (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig_medical, use_container_width=True)
        else:
            st.info("🏥 Medical attention analysis requires 'medical_attention' column")
    
    with col3:
        # Reportable incident compliance
        if 'reportable' in df_filtered.columns and 'notification_delay' in df_filtered.columns:
            reportable_incidents = df_filtered[df_filtered['reportable'] == 'Yes']
            if len(reportable_incidents) > 0:
                reportable_compliance = (reportable_incidents['notification_delay'] <= 0.5).mean() * 100
            else:
                reportable_compliance = 0
            
            fig_reportable = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reportable_compliance,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Reportable Incident Compliance (< 12 hours)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            st.plotly_chart(fig_reportable, use_container_width=True)
        else:
            st.info("📋 Reportable compliance analysis requires 'reportable' column")

elif analysis_mode == "🤖 ML Analytics":
    st.subheader("🤖 Machine Learning Analytics")
    
    # Prepare ML features
    ml_df, label_encoders = prepare_ml_features(df_filtered)
    
    # ML Analysis tabs
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["🔗 Clustering", "🚨 Anomaly Detection", "🔍 Association Rules", "📊 ML Insights"])
    
    with ml_tab1:
        st.subheader("🔗 Incident Clustering Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Clustering Parameters")
            
            clustering_method = st.selectbox(
                "Clustering Method",
                ["kmeans", "dbscan", "hierarchical"],
                help="Choose clustering algorithm"
            )
            
            if clustering_method in ['kmeans', 'hierarchical']:
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            else:
                n_clusters = None
                st.info("DBSCAN automatically determines number of clusters")
            
            if st.button("🔄 Run Clustering"):
                with st.spinner("Performing clustering analysis..."):
                    cluster_labels, metrics = perform_clustering_analysis(
                        ml_df, method=clustering_method, n_clusters=n_clusters
                    )
                    
                    if cluster_labels is not None:
                        # Store results in session state
                        st.session_state['cluster_labels'] = cluster_labels
                        st.session_state['cluster_metrics'] = metrics
                        st.success("✅ Clustering analysis completed!")
        
        with col2:
            if 'cluster_labels' in st.session_state and st.session_state['cluster_labels'] is not None:
                cluster_labels = st.session_state['cluster_labels']
                metrics = st.session_state.get('cluster_metrics', {})
                
                # Add clusters to dataframe for visualization
                viz_df = ml_df.copy()
                viz_df['cluster'] = cluster_labels
                
                # Clustering metrics
                st.markdown("### 📊 Clustering Metrics")
                if metrics:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Clusters Found", metrics.get('n_clusters', 'N/A'))
                    with col_b:
                        st.metric("Silhouette Score", f"{metrics.get('silhouette_score', 0):.3f}")
                    with col_c:
                        st.metric("Calinski Score", f"{metrics.get('calinski_score', 0):.1f}")
                
                # Cluster visualization
                if len(set(cluster_labels)) > 1:
                    # PCA for visualization
                    feature_cols = [col for col in viz_df.columns if col.endswith('_encoded')]
                    if 'age_at_incident' in viz_df.columns:
                        feature_cols.append('age_at_incident')
                    
                    if len(feature_cols) >= 2:
                        X = viz_df[feature_cols].fillna(0)
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
                        
                        fig_cluster = px.scatter(
                            x=X_pca[:, 0], y=X_pca[:, 1],
                            color=cluster_labels.astype(str),
                            title="Incident Clusters (PCA Visualization)",
                            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                        st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Cluster characteristics
                st.markdown("### 🔍 Cluster Characteristics")
                cluster_summary = []
                
                for cluster_id in sorted(set(cluster_labels)):
                    cluster_data = viz_df[viz_df['cluster'] == cluster_id]
                    
                    summary = {
                        'Cluster': f"Cluster {cluster_id}",
                        'Size': len(cluster_data),
                        'Percentage': f"{len(cluster_data)/len(viz_df)*100:.1f}%"
                    }
                    
                    # Most common characteristics
                    if 'incident_type' in cluster_data.columns:
                        most_common_type = cluster_data['incident_type'].mode()
                        summary['Common Type'] = most_common_type.iloc[0] if len(most_common_type) > 0 else 'N/A'
                    
                    if 'severity' in cluster_data.columns:
                        most_common_severity = cluster_data['severity'].mode()
                        summary['Common Severity'] = most_common_severity.iloc[0] if len(most_common_severity) > 0 else 'N/A'
                    
                    if 'age_at_incident' in cluster_data.columns:
                        summary['Avg Age'] = f"{cluster_data['age_at_incident'].mean():.1f}"
                    
                    cluster_summary.append(summary)
                
                cluster_df = pd.DataFrame(cluster_summary)
                st.dataframe(cluster_df, use_container_width=True)
    
    with ml_tab2:
        st.subheader("🚨 Anomaly Detection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Anomaly Detection Parameters")
            
            anomaly_method = st.selectbox(
                "Detection Method",
                ["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"],
                help="Choose anomaly detection algorithm"
            )
            
            contamination = st.slider(
                "Expected Contamination (%)", 
                1, 20, 10,
                help="Expected percentage of anomalies in the data"
            ) / 100
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    anomaly_labels = detect_anomalies(
                        ml_df, method=anomaly_method, contamination=contamination
                    )
                    
                    if anomaly_labels is not None:
                        st.session_state['anomaly_labels'] = anomaly_labels
                        st.success("✅ Anomaly detection completed!")
        
        with col2:
            if 'anomaly_labels' in st.session_state and st.session_state['anomaly_labels'] is not None:
                anomaly_labels = st.session_state['anomaly_labels']
                
                # Add anomaly labels to dataframe
                viz_df = ml_df.copy()
                viz_df['is_anomaly'] = (anomaly_labels == -1)
                
                # Anomaly statistics
                n_anomalies = sum(anomaly_labels == -1)
                anomaly_percentage = n_anomalies / len(anomaly_labels) * 100
                
                st.markdown("### 📊 Anomaly Statistics")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Anomalies", n_anomalies)
                with col_b:
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                with col_c:
                    st.metric("Normal Cases", len(anomaly_labels) - n_anomalies)
                
                # Visualization
                if n_anomalies > 0:
                    # PCA visualization
                    feature_cols = [col for col in viz_df.columns if col.endswith('_encoded')]
                    if 'age_at_incident' in viz_df.columns:
                        feature_cols.append('age_at_incident')
                    
                    if len(feature_cols) >= 2:
                        X = viz_df[feature_cols].fillna(0)
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
                        
                        fig_anomaly = px.scatter(
                            x=X_pca[:, 0], y=X_pca[:, 1],
                            color=viz_df['is_anomaly'].map({True: 'Anomaly', False: 'Normal'}),
                            title="Anomaly Detection Results (PCA Visualization)",
                            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
                        )
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                
                # Anomaly analysis
                if n_anomalies > 0:
                    st.markdown("### 🔍 Anomaly Analysis")
                    
                    anomalies = viz_df[viz_df['is_anomaly'] == True]
                    normal_cases = viz_df[viz_df['is_anomaly'] == False]
                    
                    # Compare distributions
                    categorical_cols = ['incident_type', 'severity', 'location']
                    
                    for col in categorical_cols:
                        if col in anomalies.columns and len(anomalies) > 0:
                            st.markdown(f"**{col} Distribution in Anomalies:**")
                            
                            anomaly_dist = anomalies[col].value_counts(normalize=True).head(5)
                            normal_dist = normal_cases[col].value_counts(normalize=True)
                            
                            comparison_data = []
                            for category in anomaly_dist.index:
                                comparison_data.append({
                                    'Category': category,
                                    'Anomaly %': anomaly_dist[category] * 100,
                                    'Normal %': normal_dist.get(category, 0) * 100
                                })
                            
                            if comparison_data:
                                comp_df = pd.DataFrame(comparison_data)
                                fig_comp = px.bar(
                                    comp_df, x='Category', y=['Anomaly %', 'Normal %'],
                                    title=f"{col} Distribution: Anomalies vs Normal",
                                    barmode='group'
                                )
                                st.plotly_chart(fig_comp, use_container_width=True)
    
    with ml_tab3:
        st.subheader("🔍 Association Rules Mining")
        
        if MLXTEND_AVAILABLE:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🎛️ Association Rules Parameters")
                
                min_support = st.slider(
                    "Minimum Support", 
                    0.01, 0.5, 0.1,
                    help="Minimum frequency of itemsets"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    0.1, 0.9, 0.6,
                    help="Minimum confidence for rules"
                )
                
                if st.button("⚡ Mine Association Rules"):
                    with st.spinner("Mining association rules..."):
                        frequent_itemsets, rules = find_association_rules(
                            ml_df, min_support=min_support, min_confidence=min_confidence
                        )
                        
                        if frequent_itemsets is not None and rules is not None:
                            st.session_state['frequent_itemsets'] = frequent_itemsets
                            st.session_state['association_rules'] = rules
                            st.success("✅ Association rules mining completed!")
                        else:
                            st.warning("No rules found with current parameters. Try lowering the thresholds.")
            
            with col2:
                if 'association_rules' in st.session_state and st.session_state['association_rules'] is not None:
                    rules = st.session_state['association_rules']
                    frequent_itemsets = st.session_state.get('frequent_itemsets', pd.DataFrame())
                    
                    st.markdown("### 📊 Association Rules Results")
                    
                    # Summary metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Frequent Itemsets", len(frequent_itemsets))
                    with col_b:
                        st.metric("Association Rules", len(rules))
                    with col_c:
                        avg_confidence = rules['confidence'].mean() if len(rules) > 0 else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    if len(rules) > 0:
                        # Top rules
                        st.markdown("### 🔝 Top Association Rules")
                        
                        # Prepare rules for display
                        display_rules = rules.copy()
                        display_rules['antecedents_str'] = display_rules['antecedents'].apply(
                            lambda x: ', '.join(list(x))
                        )
                        display_rules['consequents_str'] = display_rules['consequents'].apply(
                            lambda x: ', '.join(list(x))
                        )
                        
                        # Select top rules by confidence
                        top_rules = display_rules.nlargest(10, 'confidence')[
                            ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                        ]
                        top_rules.columns = ['If (Antecedent)', 'Then (Consequent)', 'Support', 'Confidence', 'Lift']
                        
                        st.dataframe(top_rules, use_container_width=True)
                        
                        # Rules visualization
                        fig_rules = px.scatter(
                            rules, x='support', y='confidence', 
                            size='lift', color='lift',
                            title="Association Rules: Support vs Confidence",
                            hover_data=['antecedents', 'consequents']
                        )
                        st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.warning("⚠️ Association rules mining requires the 'mlxtend' library. Install it with: pip install mlxtend")
    
    with ml_tab4:
        st.subheader("📊 ML Insights & Recommendations")
        
        # Generate ML insights
        ml_insights = []
        
        # Clustering insights
        if 'cluster_labels' in st.session_state and st.session_state['cluster_labels'] is not None:
            cluster_labels = st.session_state['cluster_labels']
            n_clusters = len(set(cluster_labels))
            ml_insights.append(f"🔗 Identified {n_clusters} distinct incident patterns through clustering analysis")
        
        # Anomaly insights
        if 'anomaly_labels' in st.session_state and st.session_state['anomaly_labels'] is not None:
            anomaly_labels = st.session_state['anomaly_labels']
            n_anomalies = sum(anomaly_labels == -1)
            anomaly_rate = n_anomalies / len(anomaly_labels) * 100
            ml_insights.append(f"🚨 Detected {n_anomalies} anomalous incidents ({anomaly_rate:.1f}% of total)")
        
        # Association rules insights
        if 'association_rules' in st.session_state and st.session_state['association_rules'] is not None:
            rules = st.session_state['association_rules']
            if len(rules) > 0:
                avg_confidence = rules['confidence'].mean()
                ml_insights.append(f"🔍 Found {len(rules)} association rules with average confidence of {avg_confidence:.1f}")
        
        # Display insights
        if ml_insights:
            st.markdown("### 🎯 Key ML Findings")
            for insight in ml_insights:
                st.markdown(f"- {insight}")
        else:
            st.info("Run the ML analyses above to generate insights and recommendations")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = [
            "🔄 **Regular Clustering**: Run clustering analysis monthly to identify new incident patterns",
            "🚨 **Anomaly Monitoring**: Set up alerts for incidents flagged as anomalies for immediate review",
            "📋 **Pattern-Based Policies**: Use association rules to update prevention protocols",
            "🎯 **Targeted Training**: Focus training on incident types identified in high-risk clusters",
            "📊 **Predictive Models**: Use clustering results to build incident prediction models"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Model performance tips
        st.markdown("### 🔧 Model Optimization Tips")
        
        tips = [
            "📈 **More Data**: Include more historical data for better pattern detection",
            "🔢 **Feature Engineering**: Add temporal features like seasonality and trends",
            "⚖️ **Parameter Tuning**: Experiment with different algorithm parameters",
            "🔄 **Cross-Validation**: Validate models using different time periods",
            "📝 **Domain Knowledge**: Incorporate NDIS-specific features and constraints"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")

# Interactive Data Table with Advanced Filtering
st.subheader("📋 Interactive Incident Explorer")

# Advanced search and filter options
col1, col2, col3 = st.columns(3)

with col1:
    search_term = st.text_input("🔍 Search descriptions/actions")

with col2:
    date_range_filter = st.date_input(
        "📅 Specific Date Range",
        value=[],
        min_value=df['incident_date'].min().date(),
        max_value=df['incident_date'].max().date()
    )

with col3:
    export_format = st.selectbox("📥 Export Format", ["CSV", "Excel", "JSON"])

# Apply additional filters
display_df = df_filtered.copy()

if search_term:
    search_cols = ['description', 'immediate_action', 'actions_taken']
    search_mask = pd.Series(False, index=display_df.index)
    for col in search_cols:
        if col in display_df.columns:
            search_mask |= display_df[col].str.contains(search_term, case=False, na=False)
    display_df = display_df[search_mask]

if len(date_range_filter) == 2:
    start_date, end_date = date_range_filter
    display_df = display_df[
        (display_df['incident_date'].dt.date >= start_date) & 
        (display_df['incident_date'].dt.date <= end_date)
    ]

# Column selection for display
available_columns = ['incident_id', 'participant_name', 'age', 'incident_date', 'incident_type',
                     'severity', 'location', 'reportable', 'notification_delay', 'medical_attention']
display_columns = st.multiselect(
    "Select columns to display",
    options=[col for col in available_columns if col in display_df.columns],
    default=[col for col in available_columns[:8] if col in display_df.columns]
)

# Display the data
if display_columns:
    # Add risk scoring to display
    if 'severity_score' not in display_columns and 'severity_score' in display_df.columns:
        display_df_show = display_df[display_columns + ['severity_score']].copy()
        display_df_show = display_df_show.sort_values(['severity_score', 'incident_date'], ascending=[False, False])
    else:
        display_df_show = display_df[display_columns].copy()
        if 'incident_date' in display_columns:
            display_df_show = display_df_show.sort_values('incident_date', ascending=False)
    
    # Style the dataframe
    if 'severity' in display_columns:
        def highlight_severity(val):
            if val == 'Critical':
                return 'background-color: #ffebee; color: #c62828; font-weight: bold'
            elif val == 'High':
                return 'background-color: #fff3e0; color: #ef6c00; font-weight: bold'
            elif val == 'Medium':
                return 'background-color: #fffde7; color: #f57f17'
            return ''
        
        styled_df = display_df_show.style.map(highlight_severity, subset=['severity'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.dataframe(display_df_show, use_container_width=True, height=400)

# Export functionality
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Data"):
        if export_format == "CSV":
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        elif export_format == "JSON":
            json_data = display_df.to_json(orient='records', date_format='iso')
            st.download_button(
                "Download JSON",
                json_data,
                f"ndis_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

with col2:
    if st.button("📊 Generate Summary Report"):
        st.subheader("📈 Executive Summary")
        
        summary_stats = {
            "Total Incidents": len(display_df),
            "Critical Incidents": len(display_df[display_df['severity'] == 'Critical']),
            "Average Notification Delay": f"{display_df['notification_delay'].mean():.1f} days" if 'notification_delay' in display_df.columns else "N/A",
            "Compliance Rate": f"{(display_df['notification_delay'] <= 1).mean() * 100:.1f}%" if 'notification_delay' in display_df.columns else "N/A",
            "Most Common Incident Type": display_df['incident_type'].mode().iloc[0] if len(display_df) > 0 else 'N/A',
            "Highest Risk Location": display_df.groupby('location')['severity_score'].mean().idxmax() if len(display_df) > 0 and 'severity_score' in display_df.columns else 'N/A'
        }
        
        for key, value in summary_stats.items():
            st.metric(key, value)

with col3:
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

# Footer with metadata
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**Records displayed:** {len(display_df)} of {len(df)}")
with col3:
    st.markdown(f"**Analysis mode:** {analysis_mode}")import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# ML Libraries for advanced analytics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Association rules libraries
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    st.warning("⚠️ mlxtend not installed. Association rules analysis will be disabled. Install with: pip install mlxtend")

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Advanced NDIS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .correlation-card {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
        animation: fadeIn 0.5s;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .alert-warning {
        background-color: #fff8e1;
        border-left-color: #ff9800;
        color: #ef6c00;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #1565c0;
    }
    .alert-success {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
        color: #2e7d32;
    }
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    .risk-low { background-color: #4caf50; }
    .risk-medium { background-color: #ff9800; }
    .risk-high { background-color: #f44336; }
    .risk-critical { background-color: #9c27b0; }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Advanced NDIS Incident Analytics Dashboard")

# Data loading functions (simplified - no demo data)
@st.cache_data
def process_data(df):
    """Process and enhance the loaded data"""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Convert date columns
        if 'incident_date' in df.columns:
            if df['incident_date'].dtype == 'object':
                df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d/%m/%Y', errors='coerce')
                if df['incident_date'].isna().all():
                    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        
        if 'notification_date' in df.columns:
            if df['notification_date'].dtype == 'object':
                df['notification_date'] = pd.to_datetime(df['notification_date'], format='%d/%m/%Y', errors='coerce')
                if df['notification_date'].isna().all():
                    df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
        else:
            # Create notification dates with some delay if missing
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(
                np.random.randint(0, 3, len(df)), unit='days'
            )
        
        # Calculate notification delay
        df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.days
        
        # Add time-based columns
        df['month'] = df['incident_date'].dt.month_name()
        df['day_of_week'] = df['incident_date'].dt.day_name()
        
        if 'incident_time' in df.columns:
            df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
        else:
            df['hour'] = np.random.randint(0, 24, len(df))
            
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        
        # Risk scoring
        severity_weights = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['severity_score'] = df['severity'].map(severity_weights).fillna(1)
        
        # Create age groups if age column exists
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        else:
            # Create a dummy age column and age groups if missing
            df['age'] = np.random.normal(35, 15, len(df)).astype(int).clip(18, 85)
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return process_data(df)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def load_local_data():
    """Try to load local data file (for development)"""
    try:
        df = pd.read_csv("/Users/darolinvinisha/PycharmProjects/MD651/Using Ollama/ndis_incidents_synthetic.csv")
        return process_data(df)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading local file: {str(e)}")
        return None

def calculate_correlations(df):
    """Calculate key correlations for analysis"""
    numeric_df = df.copy()
    
    # Convert categorical to numeric (only if columns exist)
    if 'severity' in df.columns:
        numeric_df['severity_numeric'] = numeric_df['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4})
    else:
        numeric_df['severity_numeric'] = 1
        
    if 'reportable' in df.columns:
        numeric_df['reportable_numeric'] = numeric_df['reportable'].map({'No': 0, 'Yes': 1})
    else:
        numeric_df['reportable_numeric'] = 0
        
    if 'medical_attention' in df.columns:
        numeric_df['medical_attention_numeric'] = numeric_df['medical_attention'].map({'No': 0, 'Yes': 1})
    else:
        numeric_df['medical_attention_numeric'] = 0
        
    if 'is_weekend' in df.columns:
        numeric_df['is_weekend_numeric'] = numeric_df['is_weekend'].astype(int)
    else:
        numeric_df['is_weekend_numeric'] = 0
    
    # Select only available columns for correlation
    correlation_vars = []
    possible_vars = ['age', 'severity_numeric', 'notification_delay', 'reportable_numeric', 
                    'medical_attention_numeric', 'is_weekend_numeric', 'hour']
    
    for var in possible_vars:
        if var in numeric_df.columns:
            correlation_vars.append(var)
    
    if len(correlation_vars) < 2:
        # Create minimal correlation matrix if not enough variables
        correlation_vars = ['severity_numeric', 'notification_delay']
        for var in correlation_vars:
            if var not in numeric_df.columns:
                numeric_df[var] = 1
    
    corr_matrix = numeric_df[correlation_vars].corr()
    
    return corr_matrix, numeric_df

def generate_insights(df):
    """Generate automated insights from the data"""
    insights = []
    
    try:
        # Age-related insights
        if 'age_group' in df.columns and 'severity_score' in df.columns:
            age_severity = df.groupby('age_group', observed=True)['severity_score'].mean()
            if len(age_severity) > 0:
                high_risk_age = age_severity.idxmax()
                insights.append(f"🎯 Age group '{high_risk_age}' has the highest average incident severity")
        
        # Temporal insights
        if 'is_weekend' in df.columns and 'severity_score' in df.columns:
            weekend_incidents = df[df['is_weekend']]['severity_score'].mean()
            weekday_incidents = df[~df['is_weekend']]['severity_score'].mean()
            if pd.notna(weekend_incidents) and pd.notna(weekday_incidents) and weekday_incidents > 0:
                if weekend_incidents > weekday_incidents:
                    insights.append(f"⏰ Weekend incidents are {((weekend_incidents/weekday_incidents - 1) * 100):.1f}% more severe on average")
        
        # Location insights
        if 'location' in df.columns and 'severity_score' in df.columns:
            location_risk = df.groupby('location').agg({
                'severity_score': 'mean',
                'incident_id': 'count'
            })
            if len(location_risk[location_risk['incident_id'] > 50]) > 0:
                high_risk_location = location_risk.loc[location_risk['incident_id'] > 50, 'severity_score'].idxmax()
                insights.append(f"🏢 '{high_risk_location}' shows highest severity among high-volume locations")
        
        # Reporter insights
        if 'reporter_type' in df.columns and 'notification_delay' in df.columns:
            reporter_performance = df.groupby('reporter_type').agg({
                'notification_delay': 'mean',
                'incident_id': 'count'
            })
            if len(reporter_performance) > 0:
                fastest_reporters = reporter_performance['notification_delay'].idxmin()
                insights.append(f"📞 {fastest_reporters} are the fastest at reporting incidents")
        
        # Medical attention patterns
        if 'severity' in df.columns and 'medical_attention' in df.columns:
            medical_by_severity = df.groupby('severity')['medical_attention'].apply(lambda x: (x == 'Yes').mean())
            if 'Critical' in medical_by_severity.index:
                insights.append(f"🏥 {medical_by_severity['Critical']*100:.1f}% of critical incidents require medical attention")
        
        # Default insights if no data available
        if len(insights) == 0:
            insights = [
                "📊 Data analysis in progress",
                "🔍 Exploring incident patterns",
                "📈 Building risk assessments"
            ]
            
    except Exception as e:
        insights = [f"⚠️ Analysis temporarily unavailable: {str(e)[:50]}..."]
    
    return insights

@st.cache_data
def prepare_ml_features(df):
    """Prepare features for ML analysis"""
    try:
        ml_df = df.copy()
        
        # Create derived features
        if 'incident_date' in ml_df.columns:
            ml_df['incident_year'] = ml_df['incident_date'].dt.year
            ml_df['incident_month'] = ml_df['incident_date'].dt.month
            ml_df['incident_weekday'] = ml_df['incident_date'].dt.dayofweek
        
        # Calculate age if available
        if 'age' in ml_df.columns:
            ml_df['age_at_incident'] = ml_df['age']
        else:
            ml_df['age_at_incident'] = np.random.normal(35, 15, len(ml_df)).clip(18, 85)
        
        # Encode categorical variables
        categorical_cols = ['incident_type', 'severity', 'location']
        if 'reportable' in ml_df.columns:
            categorical_cols.append('reportable')
        
        label_encoders = {}
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = ml_df[col].fillna('Unknown')
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col].astype(str))
                label_encoders[col] = le
        
        return ml_df, label_encoders
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return df, {}

def perform_clustering_analysis(df, method='kmeans', n_clusters=5):
    """Perform clustering analysis"""
    try:
        # Prepare features
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age_at_incident' in df.columns:
            feature_cols.append('age_at_incident')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'incident_month' in df.columns:
            feature_cols.append('incident_month')
        if 'incident_weekday' in df.columns:
            feature_cols.append('incident_weekday')
        
        if not feature_cols:
            st.warning("No suitable features found for clustering")
            return None, None
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate metrics
        metrics = {}
        if len(set(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
            metrics['calinski_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
            metrics['n_clusters'] = len(set(cluster_labels))
        
        return cluster_labels, metrics
        
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return None, None

def detect_anomalies(df, method='isolation_forest', contamination=0.1):
    """Detect anomalies in the data"""
    try:
        # Prepare features
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age_at_incident' in df.columns:
            feature_cols.append('age_at_incident')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        
        if not feature_cols:
            st.warning("No suitable features found for anomaly detection")
            return None
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        
        if method == 'local_outlier_factor':
            anomaly_labels = detector.fit_predict(X_scaled)
        else:
            anomaly_labels = detector.fit_predict(X_scaled)
        
        return anomaly_labels
        
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return None

def find_association_rules(df, min_support=0.1, min_confidence=0.6):
    """Find association rules in the data"""
    if not MLXTEND_AVAILABLE:
        return None, None
        
    try:
        # Prepare transaction data
        categorical_cols = ['incident_type', 'severity', 'location']
        if 'reportable' in df.columns:
            categorical_cols.append('reportable')
        
        # Create transactions
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in categorical_cols:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{col}_{row[col]}")
            if transaction:  # Only add non-empty transactions
                transactions.append(transaction)
        
        if not transactions:
            return None, None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return None, None
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return frequent_itemsets, rules
        
    except Exception as e:
        st.error(f"Association rules error: {str(e)}")
        return None, None

# Data loading UI - File Upload Only
st.sidebar.subheader("📁 Data Upload")
st.sidebar.markdown("**Upload your NDIS incidents CSV file to begin analysis**")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type="csv",
    help="Upload your NDIS incidents data in CSV format"
)

# Load data
df = None
if uploaded_file is not None:
    with st.spinner("Loading and processing your data..."):
        df = load_data_from_file(uploaded_file)
        if df is not None:
            st.sidebar.success("✅ File uploaded and processed successfully!")
        else:
            st.sidebar.error("❌ Error processing file. Please check the format.")
else:
    # Try to load local file for development (optional)
    df = load_local_data()
    if df is not None:
        st.sidebar.info("📂 Using local development data")

# Check if we have data to work with
if df is None or len(df) == 0:
    st.markdown("""
    # 🏥 NDIS Incident Analytics Dashboard
    
    ## 📁 Welcome! Please Upload Your Data
    
    To get started with your NDIS incident analysis:
    
    1. **📋 Prepare your CSV file** with incident data
    2. **📁 Use the file uploader** in the sidebar
    3. **📊 Explore your data** with advanced analytics
    
    ### 📋 Expected CSV Format:
    
    Your CSV file should include these columns (minimum required):
    - `incident_date` - Date of incident (DD/MM/YYYY format)
    - `incident_type` - Type of incident
    - `severity` - Severity level (Low, Medium, High, Critical)
    - `location` - Where the incident occurred
    
    ### 🔧 Optional Columns (for enhanced analysis):
    - `notification_date` - When incident was reported
    - `participant_name` - Participant involved
    - `age` - Participant age
    - `reportable` - Whether incident is reportable (Yes/No)
    - `medical_attention` - Medical attention required (Yes/No)
    - `reporter_type` - Type of person reporting
    - `description` - Incident description
    - `immediate_action` - Actions taken immediately
    
    ### 🎯 Features Available:
    - **📊 Executive Dashboard** - KPIs and trends
    - **🎯 Risk Analysis** - Advanced risk assessment
    - **🔗 Correlation Analysis** - Relationship discovery
    - **🔮 Predictive Analytics** - Forecasting and prediction
    - **📊 Performance Monitoring** - Compliance tracking
    - **🤖 Machine Learning** - Pattern discovery and anomaly detection
    
    **Ready to get started? Upload your file using the sidebar! 👆**
    """)
    
    # Show sample data format
    st.subheader("📋 Sample Data Format")
    sample_data = pd.DataFrame({
        'incident_date': ['01/01/2024', '02/01/2024', '03/01/2024'],
        'incident_type': ['Fall', 'Medication Error', 'Behavioral'],
        'severity': ['Medium', 'High', 'Low'],
        'location': ['Day Program', 'Residential', 'Community'],
        'reportable': ['Yes', 'No', 'No'],
        'description': ['Participant slipped', 'Wrong medication given', 'Verbal outburst']
    })
    st.dataframe(sample_data, use_container_width=True)
    
    st.stop()  # Stop execution until file is uploaded

# Continue with existing code if data is loaded
corr_matrix, numeric_df = calculate_correlations(df)
insights = generate_insights(df)

st.success(f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")

# Enhanced Sidebar with Analysis Mode
st.sidebar.header("🎛️ Advanced Controls")

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox(
    "🔬 Analysis Mode",
    ["Executive Overview", "Risk Analysis", "Correlation Explorer", "Predictive Insights", "Performance Analytics", "🤖 ML Analytics"]
)

# Interactive Date Range with Presets
st.sidebar.subheader("📅 Time Period")
preset_ranges = {
    "Last 30 Days": 30,
    "Last 90 Days": 90,
    "Last 6 Months": 180,
    "Last Year": 365,
    "All Time": None
}

time_preset = st.sidebar.selectbox("Quick Select", list(preset_ranges.keys()), index=4)

if preset_ranges[time_preset]:
    end_date = df['incident_date'].max()
    start_date = end_date - timedelta(days=preset_ranges[time_preset])
    df_filtered = df[(df['incident_date'] >= start_date) & (df['incident_date'] <= end_date)]
else:
    df_filtered = df.copy()

# Dynamic Filters
st.sidebar.subheader("🎯 Smart Filters")

# Risk-based filtering
risk_level = st.sidebar.selectbox(
    "Risk Focus",
    ["All Incidents", "High Risk Only", "Critical Only", "Repeat Participants", "High-Volume Locations"]
)

if risk_level == "High Risk Only":
    df_filtered = df_filtered[df_filtered['severity'].isin(['High', 'Critical'])]
elif risk_level == "Critical Only":
    df_filtered = df_filtered[df_filtered['severity'] == 'Critical']
elif risk_level == "Repeat Participants":
    repeat_participants = df_filtered['participant_name'].value_counts()
    repeat_names = repeat_participants[repeat_participants > 1].index
    df_filtered = df_filtered[df_filtered['participant_name'].isin(repeat_names)]
elif risk_level == "High-Volume Locations":
    location_counts = df_filtered['location'].value_counts()
    high_volume_locations = location_counts[location_counts > location_counts.quantile(0.75)].index
    df_filtered = df_filtered[df_filtered['location'].isin(high_volume_locations)]

# Multi-select filters
severity_options = df['severity'].unique()
severity_filter = st.sidebar.multiselect(
    "⚠️ Severity Level",
    options=severity_options,
    default=severity_options
)

incident_type_options = sorted(df['incident_type'].unique())
incident_type_filter = st.sidebar.multiselect(
    "📋 Incident Type",
    options=incident_type_options,
    default=incident_type_options
)

# Apply filters
df_filtered = df_filtered[
    (df_filtered['severity'].isin(severity_filter)) &
    (df_filtered['incident_type'].isin(incident_type_filter))
]

# Real-time Insights Panel
with st.sidebar:
    st.subheader("💡 Live Insights")
    for insight in insights[:3]:  # Show top 3 insights
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

# Main Dashboard Content based on Analysis Mode
if analysis_mode == "Executive Overview":
    # Enhanced KPIs with trend indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_incidents = len(df_filtered)
        prev_period_data = df[(df['incident_date'] >= df['incident_date'].max() - timedelta(days=60)) & 
                            (df['incident_date'] < df['incident_date'].max() - timedelta(days=30))]
        prev_period = len(prev_period_data)
        trend = ((total_incidents - prev_period) / prev_period * 100) if prev_period > 0 else 0
        st.metric("📊 Total Incidents", total_incidents, delta=f"{trend:+.1f}%")
    
    with col2:
        critical_count = len(df_filtered[df_filtered['severity'] == 'Critical'])
        st.metric("🚨 Critical", critical_count, delta=f"{critical_count/total_incidents*100:.1f}%" if total_incidents > 0 else "0%")
    
    with col3:
        # Check if notification_delay column exists
        if 'notification_delay' in df_filtered.columns:
            avg_delay = df_filtered['notification_delay'].mean()
            target_delay = 1.0  # Target: 1 day
            delay_status = "🟢" if avg_delay <= target_delay else "🔴"
            st.metric("⏱️ Avg Delay", f"{avg_delay:.1f}d", delta=f"{delay_status}")
        else:
            st.metric("⏱️ Avg Delay", "N/A", delta="No data")
    
    with col4:
        repeat_participants = df_filtered['participant_name'].value_counts()
        repeat_count = len(repeat_participants[repeat_participants > 1])
        st.metric("🔄 Repeat Participants", repeat_count)
    
    with col5:
        # Check if notification_delay column exists for compliance calculation
        if 'notification_delay' in df_filtered.columns:
            compliance_rate = (df_filtered['notification_delay'] <= 1).mean() * 100
            st.metric("✅ Compliance Rate", f"{compliance_rate:.1f}%")
        else:
            st.metric("✅ Compliance Rate", "N/A", delta="No data")
    
    # Interactive Incident Heatmap
    st.subheader("🔥 Incident Heatmap: Location vs Time")
    
    # Create heatmap data
    heatmap_data = df_filtered.pivot_table(
        values='incident_id', 
        index='location', 
        columns=df_filtered['incident_date'].dt.hour, 
        aggfunc='count', 
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Incident Frequency by Location and Hour of Day",
        labels=dict(x="Hour of Day", y="Location", color="Incidents"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Trend Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly trends
        monthly_trends = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
        monthly_trends.index = monthly_trends.index.astype(str)
        
        fig_trend = px.line(
            x=monthly_trends.index,
            y=monthly_trends.values,
            title="📊 Monthly Incident Trends",
            markers=True
        )
        fig_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_counts = df_filtered['severity'].value_counts()
        colors = {'Critical': '#ff4444', 'High': '#ff8800', 'Medium': '#ffcc00', 'Low': '#44ff44'}
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="⚠️ Severity Distribution",
            color=severity_counts.index,
            color_discrete_map=colors
        )
        fig_severity.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_severity, use_container_width=True)

elif analysis_mode == "Risk Analysis":
    st.subheader("🎯 Advanced Risk Analysis")
    
    # Risk Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity vs Frequency Risk Matrix
        risk_data = []
        for location in df_filtered['location'].unique():
            location_data = df_filtered[df_filtered['location'] == location]
            total_incidents = len(location_data)
            avg_severity = location_data['severity_score'].mean()
