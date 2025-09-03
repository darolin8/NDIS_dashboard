# Forecast table
        forecast_df = pd.DataFrame({
            'Month': forecast_dates.strftime('%Y-%m'),
            'Predicted Incidents': forecast.round(0).astype(int),
            'Lower Bound': forecast_lower.round(0).astype(int),
            'Upper Bound': forecast_upper.round(0).astype(int)
        })
        
        st.markdown("#### 📋 Detailed Forecast")
        st.dataframe(forecast_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Forecasting failed: {str(e)}")

# ========================================
# LOCATION RISK PROFILING
# ========================================

def location_risk_profiling(df):
    """Comprehensive location risk analysis with table and bar charts"""
    
    st.markdown("### 📍 Location Risk Profiling")
    
    if 'location' not in df.columns or 'severity' not in df.columns:
        st.warning("Location and severity columns required.")
        return
    
    # Create severity mapping
    severity_map = {'Low': 0, 'Medium': 1, 'Moderate': 1, 'High': 2, 'Critical': 2}
    df_temp = df.copy()
    df_temp['severity_numeric'] = df_temp['severity'].map(severity_map)
    
    # Calculate location statistics
    location_stats = df_temp.groupby('location').agg({
        'severity_numeric': ['count', 'mean', 'std'],
        'medical_attention_required': 'mean' if 'medical_attention_required' in df.columns else lambda x: 0,
        'reportable': 'mean' if 'reportable' in df.columns else lambda x: 0,
        'participant_id': 'nunique' if 'participant_id' in df.columns else lambda x: 1
    }).round(3)
    
    # Flatten column names
    location_stats.columns = ['Incident_Count', 'Avg_Severity', 'Severity_Std', 'Medical_Rate', 'Reportable_Rate', 'Unique_Participants']
    
    # Calculate risk score
    location_stats['Risk_Score'] = (
        location_stats['Avg_Severity'] * 0.4 +
        location_stats['Medical_Rate'] * 0.3 +
        location_stats['Reportable_Rate'] * 0.3
    )
    
    # Calculate incidents per participant (frequency metric)
    location_stats['Incidents_Per_Participant'] = (
        location_stats['Incident_Count'] / location_stats['Unique_Participants']
    ).round(2)
    
    # Sort by risk score
    location_stats = location_stats.sort_values('Risk_Score', ascending=False)
    
    # Display results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 📊 Location Risk Analysis Table")
        
        # Format table for display
        display_df = location_stats.head(20).copy()
        styled_df = display_df.style.format({
            'Avg_Severity': '{:.2f}',
            'Severity_Std': '{:.2f}',
            'Medical_Rate': '{:.1%}',
            'Reportable_Rate': '{:.1%}',
            'Risk_Score': '{:.3f}',
            'Incidents_Per_Participant': '{:.2f}'
        }).background_gradient(subset=['Risk_Score'], cmap='Reds')
        
        st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig = px.histogram(
            location_stats.reset_index(),
            x='Risk_Score',
            nbins=20,
            title="Risk Score Distribution",
            labels={'Risk_Score': 'Risk Score', 'count': 'Number of Locations'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bar charts
    st.markdown("#### 📊 Location Risk Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Top Risk Locations", "📈 Incident Volume", "🏥 Medical Attention Rate"])
    
    with tab1:
        # Top 15 risk locations
        top_risk = location_stats.head(15).reset_index()
        
        fig = px.bar(
            top_risk,
            x='Risk_Score',
            y='location',
            orientation='h',
            title="Top 15 Highest Risk Locations",
            color='Risk_Score',
            color_continuous_scale='Reds',
            labels={'Risk_Score': 'Risk Score', 'location': 'Location'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Incident volume by location
        top_volume = location_stats.sort_values('Incident_Count', ascending=False).head(15).reset_index()
        
        fig = px.bar(
            top_volume,
            x='Incident_Count',
            y='location',
            orientation='h',
            title="Top 15 Locations by Incident Volume",
            color='Incident_Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Medical attention rate by location
        high_medical = location_stats[location_stats['Medical_Rate'] > 0].sort_values('Medical_Rate', ascending=False).head(15).reset_index()
        
        if len(high_medical) > 0:
            fig = px.bar(
                high_medical,
                x='Medical_Rate',
                y='location',
                orientation='h',
                title="Top 15 Locations by Medical Attention Rate",
                color='Medical_Rate',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No locations with medical attention data available.")
    
    # Location recommendations
    st.markdown("#### 💡 Location-Based Recommendations")
    
    recommendations = []
    
    # High-risk locations
    high_risk_locations = location_stats[location_stats['Risk_Score'] > location_stats['Risk_Score'].quantile(0.8)].head(5)
    if len(high_risk_locations) > 0:
        locations_list = "', '".join(high_risk_locations.index[:3])
        recommendations.append(f"🚨 **High Priority**: Locations '{locations_list}' require immediate safety protocol review")
    
    # High volume locations
    high_volume_locations = location_stats[location_stats['Incident_Count'] >= 10].head(3)
    if len(high_volume_locations) > 0:
        recommendations.append(f"📊 **Volume Concern**: {len(high_volume_locations)} locations have ≥10 incidents - review staffing levels")
    
    # High medical attention locations
    high_medical_locations = location_stats[location_stats['Medical_Rate'] > 0.5]
    if len(high_medical_locations) > 0:
        recommendations.append(f"🏥 **Medical Resources**: {len(high_medical_locations)} locations have >50% medical attention rate - ensure medical supplies")
    
    # Safety recommendations
    recommendations.extend([
        "🛡️ **Safety Equipment**: Prioritize safety equipment installation in highest risk locations",
        "👥 **Staff Training**: Focus location-specific training on high-risk areas",
        "📋 **Regular Audits**: Monthly safety audits for top 5 risk locations",
        "📊 **Monitoring**: Implement real-time monitoring for locations with risk scores >2.0"
    ])
    
    for rec in recommendations:
        st.write(rec)
    
    return location_stats

# ========================================
# SEASONAL & TEMPORAL PATTERNS
# ========================================

def seasonal_temporal_patterns(df):
    """Detect seasonal and temporal patterns with monthly line charts"""
    
    st.markdown("### 🌍 Seasonal & Temporal Pattern Detection")
    
    if 'incident_date' not in df.columns:
        st.warning("Date column required for temporal analysis.")
        return
    
    # Prepare date data
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['year_month'] = df['incident_date'].dt.to_period('M')
    df['month_name'] = df['incident_date'].dt.month_name()
    df['day_name'] = df['incident_date'].dt.day_name()
    df['hour'] = df['incident_date'].dt.hour if 'incident_time' not in df.columns else pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
    
    # Monthly trend analysis
    st.markdown("#### 📅 Monthly Incident Trends")
    
    # Monthly data
    monthly_data = df.groupby('year_month').size()
    monthly_data.index = monthly_data.index.to_timestamp()
    
    # Calculate trend
    x_numeric = np.arange(len(monthly_data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, monthly_data.values)
    
    # Create monthly line chart
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=monthly_data.index,
        y=monthly_data.values,
        mode='lines+markers',
        name='Monthly Incidents',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Trend line
    trend_line = slope * x_numeric + intercept
    fig.add_trace(go.Scatter(
        x=monthly_data.index,
        y=trend_line,
        mode='lines',
        name=f'Trend Line (R²={r_value**2:.3f})',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Moving average
    if len(monthly_data) >= 6:
        moving_avg = monthly_data.rolling(window=3).mean()
        fig.add_trace(go.Scatter(
            x=moving_avg.index,
            y=moving_avg.values,
            mode='lines',
            name='3-Month Moving Average',
            line=dict(color='green', width=2)
        ))
    
    fig.update_layout(
        title="Monthly Incident Volume with Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    st.markdown("#### 🌤️ Seasonal Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly pattern (across all years)
        monthly_pattern = df.groupby(df['incident_date'].dt.month).size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.bar(
            x=month_names,
            y=monthly_pattern.values,
            title="Incidents by Month of Year",
            labels={'x': 'Month', 'y': 'Incidents'},
            color=monthly_pattern.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify peak months
        peak_month = monthly_pattern.idxmax()
        low_month = monthly_pattern.idxmin()
        st.info(f"Peak month: **{month_names[peak_month-1]}** ({monthly_pattern.max()} incidents)")
        st.info(f"Lowest month: **{month_names[low_month-1]}** ({monthly_pattern.min()} incidents)")
    
    with col2:
        # Day of week pattern
        day_pattern = df.groupby('day_name').size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_pattern = day_pattern.reindex(day_order)
        
        fig = px.bar(
            x=day_pattern.index,
            y=day_pattern.values,
            title="Incidents by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Incidents'},
            color=day_pattern.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekend vs weekday analysis
        weekend_incidents = day_pattern[['Saturday', 'Sunday']].sum()
        weekday_incidents = day_pattern[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].sum()
        weekend_pct = weekend_incidents / (weekend_incidents + weekday_incidents) * 100
        
        st.info(f"Weekend incidents: **{weekend_pct:.1f}%** of total")
    
    # Hourly patterns
    if df['hour'].notna().any():
        st.markdown("#### ⏰ Hourly Pattern Analysis")
        
        hourly_pattern = df.groupby('hour').size()
        
        # Create hourly heatmap-style visualization
        hours_24 = range(24)
        hourly_data = [hourly_pattern.get(h, 0) for h in hours_24]
        
        fig = go.Figure()
        
        # Bar chart for hourly distribution
        fig.add_trace(go.Bar(
            x=list(hours_24),
            y=hourly_data,
            name='Hourly Incidents',
            marker_color=hourly_data,
            marker_colorscale='Reds'
        ))
        
        fig.update_layout(
            title="Incidents by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Incidents",
            xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify peak hours
        if len(hourly_pattern) > 0:
            peak_hour = hourly_pattern.idxmax()
            low_hour = hourly_pattern.idxmin()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Hour", f"{peak_hour:02d}:00", f"{hourly_pattern.max()} incidents")
            with col2:
                st.metric("Lowest Hour", f"{low_hour:02d}:00", f"{hourly_pattern.min()} incidents")
            with col3:
                business_hours_incidents = hourly_pattern.loc[9:17].sum() if 9 in hourly_pattern.index and 17 in hourly_pattern.index else 0
                business_hours_pct = business_hours_incidents / hourly_pattern.sum() * 100
                st.metric("Business Hours %", f"{business_hours_pct:.1f}%")
    
    # Seasonal insights and recommendations
    st.markdown("#### 💡 Temporal Insights & Recommendations")
    
    insights = []
    
    # Monthly insights
    if len(monthly_data) > 6:
        recent_trend = monthly_data.tail(6).mean() - monthly_data.head(6).mean()
        if recent_trend > 0:
            insights.append(f"📈 **Increasing Trend**: Recent 6 months show {recent_trend:.1f} more incidents/month on average")
        elif recent_trend < -1:
            insights.append(f"📉 **Decreasing Trend**: Recent 6 months show {abs(recent_trend):.1f} fewer incidents/month")
        else:
            insights.append("📊 **Stable Trend**: Incident rates remain relatively stable")
    
    # Seasonal insights
    if len(monthly_pattern) >= 12:
        summer_months = monthly_pattern.loc[[12, 1, 2]].sum()  # Dec, Jan, Feb (Australian summer)
        winter_months = monthly_pattern.loc[[6, 7, 8]].sum()   # Jun, Jul, Aug (Australian winter)
        
        if summer_months > winter_months * 1.2:
            insights.append("☀️ **Summer Pattern**: Significantly more incidents during summer months")
        elif winter_months > summer_months * 1.2:
            insights.append("❄️ **Winter Pattern**: Significantly more incidents during winter months")
    
    # Day of week insights
    if len(day_pattern) == 7:
        if weekend_pct > 35:
            insights.append("🏠 **Weekend Risk**: High weekend incident rate - review weekend staffing")
        elif weekend_pct < 20:
            insights.append("📅 **Weekday Risk**: Most incidents occur during weekdays - review weekday protocols")
    
    # Hourly insights
    if df['hour'].notna().any() and len(hourly_pattern) > 10:
        night_incidents = hourly_pattern.loc[[23, 0, 1, 2, 3, 4, 5]].sum() if all(h in hourly_pattern.index for h in [23, 0, 1, 2, 3, 4, 5]) else 0
        total_incidents = hourly_pattern.sum()
        night_pct = night_incidents / total_incidents * 100
        
        if night_pct > 25:
            insights.append("🌙 **Night Risk**: High nighttime incident rate - review night shift protocols")
    
    # Recommendations
    recommendations = [
        "📊 **Monthly Monitoring**: Track monthly trends to identify seasonal staffing needs",
        "📅 **Shift Planning**: Adjust staffing based on day-of-week patterns",
        "⏰ **Time-Based Protocols**: Implement time-specific safety measures for high-risk hours",
        "🌤️ **Seasonal Preparation**: Prepare additional resources for high-risk seasons",
        "📈 **Trend Analysis**: Continue monitoring for early detection of pattern changes"
    ]
    
    all_insights = insights + recommendations
    
    for insight in all_insights:
        st.write(insight)

# ========================================
# CLUSTERING ANALYSIS
# ========================================

def clustering_analysis(X, df, feature_names):
    """Comprehensive clustering analysis with 2D and 3D PCA views"""
    
    st.markdown("### 📊 Clustering Analysis")
    
    if X is None or len(X) < 10:
        st.warning("Insufficient data for clustering analysis.")
        return
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        algorithm = st.selectbox("🔍 Clustering Algorithm", ['K-Means', 'DBSCAN', 'Hierarchical'])
    
    with col2:
        if algorithm in ['K-Means', 'Hierarchical']:
            n_clusters = st.slider("📊 Number of Clusters", 2, 10, 5)
        else:
            eps = st.slider("🎯 DBSCAN Epsilon", 0.1, 2.0, 0.5, 0.1)
    
    with col3:
        show_3d = st.checkbox("📈 3D Visualization", value=True)
    
    if st.button("🔍 Perform Clustering Analysis", type="primary"):
        with st.spinner("Performing clustering analysis..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for visualization
            pca_2d = PCA(n_components=2)
            pca_3d = PCA(n_components=3)
            
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            X_pca_3d = pca_3d.fit_transform(X_scaled)
            
            # Perform clustering
            if algorithm == 'K-Means':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif algorithm == 'DBSCAN':
                clusterer = DBSCAN(eps=eps, min_samples=5)
            elif algorithm == 'Hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            
            cluster_labels = clusterer.fit_predict(X_scaled)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                sil_score = silhouette_score(X_scaled, cluster_labels)
            else:
                sil_score = None
            
            # Add cluster labels to dataframe copy
            df_clustered = df.copy()
            df_clustered['cluster'] = cluster_labels
            df_clustered['pca_1'] = X_pca_2d[:, 0]
            df_clustered['pca_2'] = X_pca_2d[:, 1]
            df_clustered['pca_3'] = X_pca_3d[:, 2]
            
            # Display clustering results
            st.success("✅ Clustering analysis completed!")
            
            # Clustering summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                unique_clusters = len(set(cluster_labels))
                if -1 in cluster_labels:  # DBSCAN noise
                    unique_clusters -= 1
                st.metric("Number of Clusters", unique_clusters)
            
            with col2:
                if sil_score:
                    st.metric("Silhouette Score", f"{sil_score:.3f}")
                else:
                    st.metric("Silhouette Score", "N/A")
            
            with col3:
                largest_cluster = pd.Series(cluster_labels).value_counts().iloc[0] if len(cluster_labels) > 0 else 0
                st.metric("Largest Cluster Size", largest_cluster)
            
            with col4:
                if -1 in cluster_labels:
                    noise_points = (cluster_labels == -1).sum()
                    st.metric("Noise Points", noise_points)
                else:
                    st.metric("Noise Points", 0)
            
            # Visualizations
            st.markdown("#### 📈 Cluster Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["🎯 2D PCA View", "📊 3D PCA View", "📋 Cluster Analysis"])
            
            with tab1:
                # 2D PCA visualization
                color_map = {-1: 'black'}  # Noise points in black for DBSCAN
                
                fig = px.scatter(
                    x=X_pca_2d[:, 0],
                    y=X_pca_2d[:, 1],
                    color=[str(label) for label in cluster_labels],
                    title=f"2D PCA Clustering View - {algorithm}",
                    labels={'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)',
                           'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)'},
                    hover_data={
                        'Incident Date': df['incident_date'] if 'incident_date' in df.columns else None,
                        'Location': df['location'] if 'location' in df.columns else None,
                        'Severity': df['severity'] if 'severity' in df.columns else None
                    }
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if show_3d:
                    # 3D PCA visualization
                    fig = px.scatter_3d(
                        x=X_pca_3d[:, 0],
                        y=X_pca_3d[:, 1],
                        z=X_pca_3d[:, 2],
                        color=[str(label) for label in cluster_labels],
                        title=f"3D PCA Clustering View - {algorithm}",
                        labels={'x': f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
                               'y': f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
                               'z': f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})'},
                        hover_data={
                            'Incident Date': df['incident_date'] if 'incident_date' in df.columns else None,
                            'Location': df['location'] if 'location' in df.columns else None,
                            'Severity': df['severity'] if 'severity' in df.columns else None
                        }
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Cluster characteristics analysis
                st.markdown("#### 🔍 Cluster Characteristics")
                
                cluster_summary = []
                for cluster_id in sorted(set(cluster_labels)):
                    if cluster_id == -1:  # Skip noise points
                        continue
                        
                    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                    
                    if len(cluster_data) == 0:
                        continue
                    
                    summary = {
                        'Cluster': cluster_id,
                        'Size': len(cluster_data),
                        'Size %': f"{len(cluster_data)/len(df_clustered)*100:.1f}%"
                    }
                    
                    # Most common values for key columns
                    for col in ['location', 'incident_type', 'severity']:
                        if col in cluster_data.columns and len(cluster_data) > 0:
                            most_common = cluster_data[col].mode()
                            if len(most_common) > 0:
                                summary[f'Most Common {col.title()}'] = most_common.iloc[0]
                    
                    # Average values for boolean columns
                    for col in ['medical_attention_required', 'reportable']:
                        if col in cluster_data.columns:
                            avg_val = cluster_data[col].mean()
                            summary[f'Avg {col.replace("_", " ").title()}'] = f"{avg_val:.1%}"
                    
                    # Time patterns
                    if 'hour' in cluster_data.columns:
                        avg_hour = cluster_data['hour'].mean()
                        summary['Avg Hour'] = f"{avg_hour:.1f}"
                    
                    cluster_summary.append(summary)
                
                if cluster_summary:
                    cluster_df = pd.DataFrame(cluster_summary)
                    st.dataframe(cluster_df, use_container_width=True)
                
                # PCA component analysis
                st.markdown("#### 🎯 PCA Component Analysis")
                
                # Show which original features contribute most to each PC
                components_df = pd.DataFrame(
                    pca_3d.components_[:3].T,
                    columns=['PC1', 'PC2', 'PC3'],
                    index=feature_names
                )
                
                # Get top contributors for each PC
                for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
                    st.write(f"**{pc} Top Contributors:**")
                    top_contributors = components_df[pc].abs().nlargest(5)
                    for feature, contribution in top_contributors.items():
                        direction = "+" if components_df.loc[feature, pc] > 0 else "-"
                        st.write(f"  {direction} {feature}: {abs(contribution):.3f}")
            
            # Clustering insights
            st.markdown("#### 💡 Clustering Insights")
            
            insights = []
            
            # Cluster size distribution
            cluster_sizes = pd.Series(cluster_labels).value_counts()
            if len(cluster_sizes) > 1:
                largest_pct = cluster_sizes.iloc[0] / len(cluster_labels) * 100
                if largest_pct > 70:
                    insights.append(f"📊 **Dominant Pattern**: One cluster contains {largest_pct:.1f}% of incidents - indicates strong common pattern")
                elif largest_pct < 30 and len(cluster_sizes) >= 4:
                    insights.append("📊 **Diverse Patterns**: Well-distributed clusters indicate diverse incident patterns")
            
            # Silhouette score interpretation
            if sil_score:
                if sil_score > 0.7:
                    insights.append(f"✅ **Excellent Clustering**: Silhouette score {sil_score:.3f} indicates well-separated clusters")
                elif sil_score > 0.5:
                    insights.append(f"👍 **Good Clustering**: Silhouette score {sil_score:.3f} indicates reasonable cluster separation")
                elif sil_score > 0.25:
                    insights.append(f"⚠️ **Moderate Clustering**: Silhouette score {sil_score:.3f} indicates some overlap between clusters")
                else:
                    insights.append(f"❌ **Poor Clustering**: Silhouette score {sil_score:.3f} indicates poorly separated clusters")
            
            # Actionable recommendations
            recommendations = [
                "🎯 **Targeted Interventions**: Design specific strategies for each cluster's characteristics",
                "📊 **Resource Allocation**: Allocate resources based on cluster size and risk patterns",
                "🔄 **Pattern Monitoring**: Monitor if incidents shift between cluster patterns over time",
                "📋 **Policy Development**: Develop cluster-specific policies and procedures"
            ]
            
            all_insights = insights + recommendations
            for insight in all_insights:
                st.write(insight)

# ========================================
# CORRELATION ANALYSIS
# ========================================

def correlation_analysis(X, feature_names, df):
    """Comprehensive correlation analysis with heatmap and insights"""
    
    st.markdown("### 🔗 Correlation Analysis")
    
    if X is None or len(feature_names) < 3:
        st.warning("Insufficient features for correlation analysis.")
        return
    
    # Create correlation matrix
    correlation_matrix = X.corr()
    
    # Correlation heatmap
    st.markdown("#### 🎯 Feature Correlation Heatmap")
    
    # Interactive plotly heatmap
    fig = px.imshow(
        correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title="Feature Correlation Matrix",
        zmin=-1,
        zmax=1
    """
COMPREHENSIVE NDIS ANALYTICS SYSTEM
Participants & Carers Predictive Analytics with Advanced Visualizations

Features:
- Predictive Models Comparison with Feature Importance & Confusion Matrix
- Incident Volume Forecasting
- Location Risk Profiling (Table + Bar Charts)
- Seasonal & Temporal Pattern Detection
- Clustering Analysis (2D & 3D PCA)
- Correlation Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, auc, silhouette_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

# Time Series
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
from datetime import datetime, timedelta

# Page Config
st.set_page_config(
    page_title="NDIS Advanced Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# COMPREHENSIVE FEATURE ENGINEERING
# ========================================

@st.cache_data
def create_comprehensive_features(df):
    """Create comprehensive features for participants and carers"""
    
    if df.empty:
        return None, None, None
    
    st.info("🔧 Creating comprehensive participant and carer features...")
    
    features_df = df.copy()
    
    # Ensure required datetime columns
    if 'incident_date' in features_df.columns:
        features_df['incident_date'] = pd.to_datetime(features_df['incident_date'])
        
    if 'incident_time' in features_df.columns:
        features_df['incident_time'] = pd.to_datetime(features_df['incident_time'], format='%H:%M', errors='coerce')
        
    # Create datetime combination
    if 'incident_date' in features_df.columns and 'incident_time' in features_df.columns:
        features_df['incident_datetime'] = pd.to_datetime(
            features_df['incident_date'].dt.strftime('%Y-%m-%d') + ' ' + 
            features_df['incident_time'].dt.strftime('%H:%M')
        )
    elif 'incident_date' in features_df.columns:
        features_df['incident_datetime'] = features_df['incident_date']
    
    # Sort by datetime for sequential features
    if 'incident_datetime' in features_df.columns:
        features_df = features_df.sort_values('incident_datetime').reset_index(drop=True)
    
    # ===== TEMPORAL FEATURES =====
    if 'incident_datetime' in features_df.columns:
        features_df['hour'] = features_df['incident_datetime'].dt.hour
        features_df['day_of_week'] = features_df['incident_datetime'].dt.dayofweek
        features_df['day_of_year'] = features_df['incident_datetime'].dt.dayofyear
        features_df['month'] = features_df['incident_datetime'].dt.month
        features_df['quarter'] = features_df['incident_datetime'].dt.quarter
        features_df['week_of_year'] = features_df['incident_datetime'].dt.isocalendar().week
        
        # Time period flags
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_early_morning'] = ((features_df['hour'] >= 5) & (features_df['hour'] <= 8)).astype(int)
        features_df['is_morning'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 11)).astype(int)
        features_df['is_afternoon'] = ((features_df['hour'] >= 12) & (features_df['hour'] <= 17)).astype(int)
        features_df['is_evening'] = ((features_df['hour'] >= 18) & (features_df['hour'] <= 22)).astype(int)
        features_df['is_night'] = ((features_df['hour'] >= 23) | (features_df['hour'] <= 4)).astype(int)
        features_df['is_business_hours'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 17)).astype(int)
        
        # Seasonal features
        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        features_df['season'] = features_df['month'].map(season_map)
        features_df['is_summer'] = (features_df['season'] == 0).astype(int)
        features_df['is_autumn'] = (features_df['season'] == 1).astype(int)
        features_df['is_winter'] = (features_df['season'] == 2).astype(int)
        features_df['is_spring'] = (features_df['season'] == 3).astype(int)
    
    # ===== LOCATION FEATURES =====
    if 'location' in features_df.columns:
        # Location type flags
        features_df['is_participant_home'] = features_df['location'].str.contains('participant home|own home|home', case=False, na=False).astype(int)
        features_df['is_kitchen'] = features_df['location'].str.contains('kitchen', case=False, na=False).astype(int)
        features_df['is_bathroom'] = features_df['location'].str.contains('bathroom|toilet', case=False, na=False).astype(int)
        features_df['is_bedroom'] = features_df['location'].str.contains('bedroom', case=False, na=False).astype(int)
        features_df['is_living_room'] = features_df['location'].str.contains('living', case=False, na=False).astype(int)
        features_df['is_activity_room'] = features_df['location'].str.contains('activity|day room', case=False, na=False).astype(int)
        features_df['is_outdoor'] = features_df['location'].str.contains('outdoor|garden|yard|park', case=False, na=False).astype(int)
        features_df['is_transport'] = features_df['location'].str.contains('transport|car|vehicle|road', case=False, na=False).astype(int)
        features_df['is_community'] = features_df['location'].str.contains('community|shop|store|mall', case=False, na=False).astype(int)
        features_df['is_medical_facility'] = features_df['location'].str.contains('hospital|clinic|medical', case=False, na=False).astype(int)
        
        # Location risk scoring
        def calculate_location_risk_score(row):
            risk_score = 1  # Base risk
            if row['is_kitchen'] or row['is_bathroom']: risk_score += 3
            elif row['is_transport'] or row['is_medical_facility']: risk_score += 2
            elif row['is_activity_room'] or row['is_community']: risk_score += 1
            elif row['is_participant_home'] and not (row['is_kitchen'] or row['is_bathroom']): risk_score = max(1, risk_score - 1)
            return min(risk_score, 5)
        
        features_df['location_risk_score'] = features_df.apply(calculate_location_risk_score, axis=1)
    
    # ===== PARTICIPANT FEATURES =====
    if 'participant_id' in features_df.columns:
        # Sort by participant and datetime for sequential features
        features_df = features_df.sort_values(['participant_id', 'incident_datetime']).reset_index(drop=True)
        
        # Participant incident history
        features_df['participant_incident_count'] = features_df.groupby('participant_id').cumcount() + 1
        
        if 'incident_datetime' in features_df.columns:
            features_df['days_since_last_incident'] = (
                features_df.groupby('participant_id')['incident_datetime']
                .diff().dt.days.fillna(999)
            )
        
        # Participant risk profiling (calculated up to current incident)
        participant_stats = []
        for idx, row in features_df.iterrows():
            participant_history = features_df[
                (features_df['participant_id'] == row['participant_id']) & 
                (features_df.index < idx)
            ]
            
            if len(participant_history) > 0:
                # Medical attention rate
                med_rate = participant_history['medical_attention_required'].mean() if 'medical_attention_required' in participant_history.columns else 0
                
                # High severity rate
                if 'severity' in participant_history.columns:
                    severity_map = {'Low': 0, 'Medium': 1, 'Moderate': 1, 'High': 2, 'Critical': 2}
                    severity_numeric = participant_history['severity'].map(severity_map)
                    high_severity_rate = (severity_numeric >= 2).mean()
                else:
                    high_severity_rate = 0
                
                # Reportable rate
                reportable_rate = participant_history['reportable'].mean() if 'reportable' in participant_history.columns else 0
                
                # Average days between incidents
                avg_days_between = participant_history['days_since_last_incident'].mean() if len(participant_history) > 1 else 999
            else:
                med_rate = 0
                high_severity_rate = 0
                reportable_rate = 0
                avg_days_between = 999
            
            participant_stats.append({
                'participant_medical_rate': med_rate,
                'participant_high_severity_rate': high_severity_rate,
                'participant_reportable_rate': reportable_rate,
                'participant_avg_days_between_incidents': avg_days_between
            })
        
        # Add participant statistics
        for key in participant_stats[0].keys():
            features_df[key] = [stat[key] for stat in participant_stats]
        
        # High-risk participant flags
        if len(features_df) > 20:
            features_df['is_high_risk_participant'] = (
                features_df['participant_incident_count'] >= features_df['participant_incident_count'].quantile(0.8)
            ).astype(int)
        else:
            features_df['is_high_risk_participant'] = 0
    
    # ===== CARER FEATURES =====
    if 'carer_id' in features_df.columns:
        # Carer incident history
        features_df['carer_incident_count'] = features_df.groupby('carer_id').cumcount() + 1
        
        if 'incident_datetime' in features_df.columns:
            features_df['carer_days_since_last_incident'] = (
                features_df.groupby('carer_id')['incident_datetime']
                .diff().dt.days.fillna(999)
            )
        
        # Carer-participant combinations
        features_df['carer_participant_incident_count'] = (
            features_df.groupby(['carer_id', 'participant_id']).cumcount() + 1
        )
        
        # Same incident type count for carer
        if 'incident_type' in features_df.columns:
            features_df['carer_same_incident_type_count'] = (
                features_df.groupby(['carer_id', 'incident_type']).cumcount() + 1
            )
        
        # Carer performance metrics (calculated up to current incident)
        carer_stats = []
        for idx, row in features_df.iterrows():
            carer_history = features_df[
                (features_df['carer_id'] == row['carer_id']) & 
                (features_df.index < idx)
            ]
            
            if len(carer_history) > 0:
                # Carer medical attention rate
                carer_med_rate = carer_history['medical_attention_required'].mean() if 'medical_attention_required' in carer_history.columns else 0
                
                # Carer high severity rate
                if 'severity' in carer_history.columns:
                    severity_map = {'Low': 0, 'Medium': 1, 'Moderate': 1, 'High': 2, 'Critical': 2}
                    carer_severity_numeric = carer_history['severity'].map(severity_map)
                    carer_high_severity_rate = (carer_severity_numeric >= 2).mean()
                else:
                    carer_high_severity_rate = 0
                
                # Carer reportable rate
                carer_reportable_rate = carer_history['reportable'].mean() if 'reportable' in carer_history.columns else 0
            else:
                carer_med_rate = 0
                carer_high_severity_rate = 0
                carer_reportable_rate = 0
            
            carer_stats.append({
                'carer_medical_rate': carer_med_rate,
                'carer_high_severity_rate': carer_high_severity_rate,
                'carer_reportable_rate': carer_reportable_rate
            })
        
        # Add carer statistics
        for key in carer_stats[0].keys():
            features_df[key] = [stat[key] for stat in carer_stats]
        
        # High-risk carer flags
        if len(features_df) > 20:
            features_df['is_high_risk_carer'] = (
                features_df['carer_incident_count'] >= features_df['carer_incident_count'].quantile(0.8)
            ).astype(int)
        else:
            features_df['is_high_risk_carer'] = 0
    
    # ===== CONTEXTUAL FEATURES =====
    # High-risk combinations
    features_df['high_risk_time_location'] = (
        ((features_df['is_early_morning'] == 1) | (features_df['is_night'] == 1)) & 
        ((features_df['is_kitchen'] == 1) | (features_df['is_bathroom'] == 1))
    ).astype(int)
    
    if 'participant_id' in features_df.columns and 'carer_id' in features_df.columns:
        features_df['high_risk_participant_carer'] = (
            (features_df['is_high_risk_participant'] == 1) & 
            (features_df['is_high_risk_carer'] == 1)
        ).astype(int)
    
    # ===== ENCODED CATEGORICAL FEATURES =====
    label_encoders = {}
    categorical_columns = ['location', 'incident_type', 'severity']
    
    # Add carer and participant if they exist
    if 'carer_id' in features_df.columns:
        categorical_columns.append('carer_id')
    if 'participant_id' in features_df.columns:
        categorical_columns.append('participant_id')
    
    for col in categorical_columns:
        if col in features_df.columns:
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].fillna('Unknown'))
            label_encoders[col] = le
    
    # ===== FEATURE SELECTION =====
    feature_columns = [
        # Temporal features
        'hour', 'day_of_week', 'month', 'quarter', 'season',
        'is_weekend', 'is_early_morning', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'is_business_hours', 'is_summer', 'is_autumn', 'is_winter', 'is_spring',
        
        # Location features
        'is_participant_home', 'is_kitchen', 'is_bathroom', 'is_bedroom', 'is_living_room',
        'is_activity_room', 'is_outdoor', 'is_transport', 'is_community', 'is_medical_facility',
        'location_risk_score',
        
        # Contextual features
        'high_risk_time_location'
    ]
    
    # Add participant features if available
    if 'participant_id' in features_df.columns:
        feature_columns.extend([
            'participant_incident_count', 'days_since_last_incident',
            'participant_medical_rate', 'participant_high_severity_rate', 'participant_reportable_rate',
            'participant_avg_days_between_incidents', 'is_high_risk_participant'
        ])
    
    # Add carer features if available
    if 'carer_id' in features_df.columns:
        feature_columns.extend([
            'carer_incident_count', 'carer_days_since_last_incident',
            'carer_participant_incident_count', 'carer_medical_rate', 'carer_high_severity_rate',
            'carer_reportable_rate', 'is_high_risk_carer'
        ])
        
        if 'incident_type' in features_df.columns:
            feature_columns.append('carer_same_incident_type_count')
        
        if 'participant_id' in features_df.columns:
            feature_columns.append('high_risk_participant_carer')
    
    # Add encoded features
    for col in categorical_columns:
        if f'{col}_encoded' in features_df.columns:
            feature_columns.append(f'{col}_encoded')
    
    # Filter to existing columns
    feature_columns = [col for col in feature_columns if col in features_df.columns]
    
    # Prepare final feature matrix
    X = features_df[feature_columns].fillna(0)
    
    st.success(f"✅ Created {len(feature_columns)} comprehensive features!")
    
    return X, feature_columns, features_df

# ========================================
# PREDICTIVE MODELS COMPARISON
# ========================================

def predictive_models_comparison(X, y, feature_names, target_name="severity"):
    """Comprehensive model comparison with feature importance and confusion matrix"""
    
    st.markdown("### 🤖 Predictive Models Comparison")
    
    if X is None or len(X) < 20:
        st.warning("Insufficient data for model training.")
        return None, None
    
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    # Train models and collect results
    results = {}
    model_performances = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Model Performance Comparison")
        
        for name, model in models.items():
            try:
                # Train model
                if name == 'Logistic Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # AUC for binary classification
                if len(np.unique(y)) == 2:
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc_score = None
                
                model_performances.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'CV Mean': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'AUC': auc_score if auc_score else 'N/A'
                })
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy,
                    'feature_importance': getattr(model, 'feature_importances_', None)
                }
                
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
    
    with col2:
        # Performance metrics table
        if model_performances:
            performance_df = pd.DataFrame(model_performances)
            st.dataframe(
                performance_df.style.format({
                    'Accuracy': '{:.3f}',
                    'CV Mean': '{:.3f}',
                    'CV Std': '{:.3f}'
                }).background_gradient(subset=['Accuracy'], cmap='Greens'),
                use_container_width=True
            )
    
    # Feature Importance Analysis
    if results and any(result.get('feature_importance') is not None for result in results.values()):
        st.markdown("#### 🎯 Feature Importance Analysis")
        
        # Get feature importance from Random Forest
        rf_importance = results.get('Random Forest', {}).get('feature_importance')
        
        if rf_importance is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 15 features
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_importance
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_df.sort_values('Importance'), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Importance (Random Forest)",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature categories
                feature_categories = {
                    'Temporal': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_morning', 'is_evening', 'season'],
                    'Location': ['is_kitchen', 'is_bathroom', 'is_home', 'location_risk_score'],
                    'Participant': ['participant_incident_count', 'participant_medical_rate', 'is_high_risk_participant'],
                    'Carer': ['carer_incident_count', 'carer_medical_rate', 'is_high_risk_carer'],
                    'Contextual': ['high_risk_time_location', 'high_risk_participant_carer']
                }
                
                category_importance = {}
                for category, features in feature_categories.items():
                    category_importance[category] = sum(
                        importance_df[importance_df['Feature'].str.contains('|'.join(features), case=False, na=False)]['Importance']
                    )
                
                category_df = pd.DataFrame(
                    list(category_importance.items()), 
                    columns=['Category', 'Total_Importance']
                ).sort_values('Total_Importance', ascending=False)
                
                fig = px.pie(
                    category_df, 
                    values='Total_Importance', 
                    names='Category',
                    title="Feature Importance by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    if results:
        st.markdown("#### 🎭 Confusion Matrices")
        
        num_models = len(results)
        cols = st.columns(min(num_models, 2))
        
        for idx, (name, result) in enumerate(results.items()):
            col_idx = idx % 2
            
            with cols[col_idx]:
                # Create confusion matrix
                cm = confusion_matrix(y_test, result['predictions'])
                
                # Convert to DataFrame for better visualization
                if len(np.unique(y)) <= 5:  # Only show for manageable number of classes
                    labels = sorted(np.unique(y))
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[str(l) for l in labels],
                        y=[str(l) for l in labels],
                        title=f"{name} - Confusion Matrix",
                        color_continuous_scale='Blues',
                        text_auto=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    return results, model_performances

# ========================================
# INCIDENT VOLUME FORECASTING
# ========================================

def incident_volume_forecasting(df, periods=6):
    """Time series forecasting of incident volume"""
    
    st.markdown("### 📈 Incident Volume Forecasting")
    
    if df.empty or 'incident_date' not in df.columns:
        st.warning("Date column required for forecasting.")
        return
    
    try:
        # Prepare time series data
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        df_monthly = df.groupby(df['incident_date'].dt.to_period('M')).size()
        df_monthly.index = df_monthly.index.to_timestamp()
        
        if len(df_monthly) < 6:
            st.warning("Need at least 6 months of data for reliable forecasting.")
            return
        
        # Create forecast
        model = ExponentialSmoothing(df_monthly, trend='add', seasonal=None)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)
        
        # Create forecast dates
        last_date = df_monthly.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(),
            periods=periods,
            freq='M'
        )
        
        # Visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_monthly.index,
            y=df_monthly.values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=3)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast.values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        # Add confidence intervals (simplified)
        forecast_upper = forecast * 1.1
        forecast_lower = forecast * 0.9
        
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(forecast_upper) + list(forecast_lower[::-1]),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"Incident Volume Forecast - Next {periods} Months",
            xaxis_title="Date",
            yaxis_title="Number of Incidents",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_avg = df_monthly.tail(6).mean()
            st.metric("Current 6-Month Avg", f"{current_avg:.1f}")
        
        with col2:
            forecast_avg = forecast.mean()
            st.metric("Forecast 6-Month Avg", f"{forecast_avg:.1f}")
        
        with col3:
            trend = ((forecast_avg - current_avg) / current_avg * 100) if current_avg > 0 else 0
            st.metric("Trend", f"{trend:+.1f}%")
        
        # Forecast table
        forecast_df = pd.DataFrame({
            'Month': forecast_dates.strftime('%Y-%m'),
            'Predicted Incidents': forecast.round(0).astype(int),
            'Lower Bound': forecast_lower.round(0
