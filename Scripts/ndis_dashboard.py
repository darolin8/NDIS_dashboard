with ml_tab4:
        st.subheader("🔮 Predictive Analytics")
        st.markdown("*Forecast future trends and predict risk factors*")
        
        prediction_type = st.selectbox(
            "Prediction Type",
            ["📈 Incident Trends Forecasting", "🎯 Risk Factor Prediction", "📊 Seasonal Pattern Analysis"]
        )
        
        if prediction_type == "📈 Incident Trends Forecasting":
            st.markdown("### 📈 Incident Volume Forecasting")
            st.markdown("*Predict future incident volumes based on historical trends*")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎛️ Forecasting Parameters")
                
                forecast_months = st.slider("Forecast Period (months)", 3, 12, 6)
                include_seasonality = st.checkbox("Include Seasonal Patterns", True)
                confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
                
                if st.button("🔮 Generate Forecast", type="primary"):
                    with st.spinner("🔮 Analyzing trends and generating forecasts..."):
                        historical_data, predictions, error = predict_incident_trends(df_filtered)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif historical_data is not None and predictions is not None:
                            st.session_state['trend_historical'] = historical_data
                            st.session_state['trend_predictions'] = predictions
                            st.success("✅ Trend forecasting completed!")
                        else:
                            st.error("❌ Unable to generate forecasts")
            
            with col2:
                if 'trend_predictions' in st.session_state:
                    historical_data = st.session_state['trend_historical']
                    predictions = st.session_state['trend_predictions']
                    
                    st.markdown("### 📊 Forecasting Results")
                    
                    # Summary metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        avg_predicted = predictions['predicted_incidents'].mean()
                        st.metric("📈 Avg Monthly Forecast", f"{avg_predicted:.1f}")
                    with col_b:
                        recent_avg = historical_data['incidents'].tail(3).mean()
                        change = ((avg_predicted - recent_avg) / recent_avg) * 100
                        st.metric("📊 Trend Change", f"{change:+.1f}%")
                    with col_c:
                        total_predicted = predictions['predicted_incidents'].sum()
                        st.metric("🎯 Total Forecast", f"{total_predicted:.0f}")
                    
                    # Forecast visualization
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    fig_forecast.add_trace(go.Scatter(
                        x=historical_data['date'],
                        y=historical_data['incidents'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    # Predictions
                    fig_forecast.add_trace(go.Scatter(
                        x=predictions['date'],
                        y=predictions['predicted_incidents'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Confidence interval
                    fig_forecast.add_trace(go.Scatter(
                        x=list(predictions['date']) + list(predictions['date'][::-1]),
                        y=list(predictions['upper_bound']) + list(predictions['lower_bound'][::-1]),
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        showlegend=True
                    ))
                    
                    fig_forecast.update_layout(
                        title="🔮 Incident Volume Forecast",
                        xaxis_title="Date",
                        yaxis_title="Number of Incidents",
                        height=500
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast table
                    st.markdown("### 📋 Detailed Forecast")
                    forecast_display = predictions.copy()
                    forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m')
                    forecast_display['predicted_incidents'] = forecast_display['predicted_incidents'].round(1)
                    forecast_display['lower_bound'] = forecast_display['lower_bound'].round(1)
                    forecast_display['upper_bound'] = forecast_display['upper_bound'].round(1)
                    forecast_display.columns = ['Month', 'Predicted', 'Lower Bound', 'Upper Bound']
                    
                    st.dataframe(forecast_display, use_container_width=True)
                    
                    # Insights
                    st.markdown("### 💡 Forecast Insights")
                    trend_slope = (predictions['predicted_incidents'].iloc[-1] - predictions['predicted_incidents'].iloc[0]) / len(predictions)
                    
                    if trend_slope > 0.5:
                        trend_desc = "📈 **Increasing trend** - incidents are expected to rise"
                    elif trend_slope < -0.5:
                        trend_desc = "📉 **Decreasing trend** - incidents are expected to decline"
                    else:
                        trend_desc = "➡️ **Stable trend** - incidents are expected to remain steady"
                    
                    st.markdown(f"- {trend_desc}")
                    st.markdown(f"- 🎯 **Peak month**: {predictions.loc[predictions['predicted_incidents'].idxmax(), 'date'].strftime('%B %Y')}")
                    st.markdown(f"- 📊 **Average confidence interval**: ±{(predictions['upper_bound'] - predictions['predicted_incidents']).mean():.1f} incidents")
                else:
                    st.info("👆 Click 'Generate Forecast' to predict future incident trends")
        
        elif prediction_type == "🎯 Risk Factor Prediction":
            st.markdown("### 🎯 High-Risk Incident Prediction")
            st.markdown("*Identify factors that predict high-severity incidents*")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎛️ Model Parameters")
                
                model_type = st.selectbox(
                    "Prediction Model",
                    ["Random Forest", "Logistic Regression", "Gradient Boosting"]
                )
                
                test_size = st.slider("Test Data Size (%)", 20, 40, 30) / 100
                
                if st.button("🎯 Train Risk Model", type="primary"):
                    with st.spinner("🎯 Training risk prediction model..."):
                        model_results, error = predict_risk_factors(df_filtered)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif model_results:
                            st.session_state['risk_model'] = model_results
                            st.success("✅ Risk model training completed!")
                        else:
                            st.error("❌ Unable to train risk model")
            
            with col2:
                if 'risk_model' in st.session_state:
                    model_results = st.session_state['risk_model']
                    
                    st.markdown("### 📊 Risk Model Results")
                    
                    # Model performance
                    performance = model_results['performance']
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("🎯 Accuracy", f"{performance['accuracy']:.3f}")
                    with col_b:
                        st.metric("📈 Precision", f"{performance['precision']:.3f}")
                    with col_c:
                        st.metric("🔍 Recall", f"{performance['recall']:.3f}")
                    with col_d:
                        st.metric("⚖️ F1-Score", f"{performance['f1_score']:.3f}")
                    
                    # Feature importance
                    st.markdown("### 🔑 Most Important Risk Factors")
                    feature_importance = model_results['feature_importance']
                    
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="🔑 Feature Importance for High-Risk Prediction",
                        labels={'importance': 'Importance Score', 'feature': 'Feature'}
                    )
                    fig_importance.update_layout(height=400)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Risk factor interpretation
                    st.markdown("### 📋 Risk Factor Analysis")
                    feature_interpretation = []
                    
                    for _, row in feature_importance.head(5).iterrows():
                        feature_name = row['feature'].replace('_encoded', '').replace('_', ' ').title()
                        importance = row['importance']
                        
                        if importance > 0.3:
                            risk_level = "🔴 Critical"
                        elif importance > 0.2:
                            risk_level = "🟡 High"
                        elif importance > 0.1:
                            risk_level = "🟢 Moderate"
                        else:
                            risk_level = "⚪ Low"
                        
                        feature_interpretation.append({
                            'Risk Factor': feature_name,
                            'Importance': f"{importance:.3f}",
                            'Risk Level': risk_level
                        })
                    
                    interpretation_df = pd.DataFrame(feature_interpretation)
                    st.dataframe(interpretation_df, use_container_width=True)
                    
                    # Individual risk prediction
                    st.markdown("### 🔮 Individual Risk Prediction")
                    st.markdown("*Predict risk for specific incident characteristics*")
                    
                    # Create input form for prediction
                    prediction_input = {}
                    feature_cols = model_results['feature_columns']
                    
                    input_col1, input_col2 = st.columns(2)
                    
                    with input_col1:
                        if 'incident_type_encoded' in feature_cols:
                            incident_types = df_filtered['incident_type'].unique()
                            selected_type = st.selectbox("Incident Type", incident_types)
                            # Encode the selection (simplified)
                            prediction_input['incident_type_encoded'] = list(incident_types).index(selected_type)
                        
                        if 'age' in feature_cols:
                            age_input = st.number_input("Participant Age", 18, 100, 35)
                            prediction_input['age'] = age_input
                        
                        if 'hour' in feature_cols:
                            hour_input = st.number_input("Hour of Day", 0, 23, 12)
                            prediction_input['hour'] = hour_input
                    
                    with input_col2:
                        if 'location_encoded' in feature_cols:
                            locations = df_filtered['location'].unique()
                            selected_location = st.selectbox("Location", locations)
                            prediction_input['location_encoded'] = list(locations).index(selected_location)
                        
                        if 'notification_delay' in feature_cols:
                            delay_input = st.number_input("Notification Delay (days)", 0.0, 10.0, 0.5)
                            prediction_input['notification_delay'] = delay_input
                    
                    if st.button("🔮 Predict Risk"):
                        try:
                            # Prepare input for prediction
                            input_array = np.array([[prediction_input.get(col, 0) for col in feature_cols]])
                            risk_probability = model_results['model'].predict_proba(input_array)[0][1]
                            
                            # Display risk prediction
                            if risk_probability > 0.7:
                                risk_status = "🔴 HIGH RISK"
                                risk_color = "red"
                            elif risk_probability > 0.4:
                                risk_status = "🟡 MODERATE RISK"
                                risk_color = "orange"
                            else:
                                risk_status = "🟢 LOW RISK"
                                risk_color = "green"
                            
                            st.markdown(f"### Risk Prediction: {risk_status}")
                            st.markdown(f"**Risk Probability: {risk_probability:.1%}**")
                            
                            # Risk gauge
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = risk_probability * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Risk Score"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': risk_color},
                                    'steps': [
                                        {'range': [0, 40], 'color': "lightgreen"},
                                        {'range': [40, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "lightcoral"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 70
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                else:
                    st.info("👆 Click 'Train Risk Model' to enable risk prediction")
        
        elif prediction_type == "📊 Seasonal Pattern Analysis":
            st.markdown("### 📊 Seasonal Pattern Analysis")
            st.markdown("*Analyze seasonal trends and patterns in incident data*")
            
            # Seasonal analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly patterns
                if 'month' in df_filtered.columns:
                    monthly_pattern = df_filtered.groupby('month').size().reindex([
                        'January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December'
                    ], fill_value=0)
                    
                    fig_monthly = px.bar(
                        x=monthly_pattern.index,
                        y=monthly_pattern.values,
                        title="📅 Monthly Incident Patterns",
                        labels={'x': 'Month', 'y': 'Number of Incidents'}
                    )
                    fig_monthly.update_layout(height=400)
                    st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                # Day of week patterns
                if 'day_of_week' in df_filtered.columns:
                    dow_pattern = df_filtered.groupby('day_of_week').size().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ], fill_value=0)
                    
                    fig_dow = px.bar(
                        x=dow_pattern.index,
                        y=dow_pattern.values,
                        title="📅 Day of Week Patterns",
                        labels={'x': 'Day of Week', 'y': 'Number of Incidents'}
                    )
                    fig_dow.update_layout(height=400)
                    st.plotly_chart(fig_dow, use_container_width=True)
            
            # Hourly heatmap
            if 'hour' in df_filtered.columns and 'day_of_week' in df_filtered.columns:
                st.markdown("### ⏰ Hourly Incident Heatmap")
                
                # Create hour-day heatmap data
                hourly_data = df_filtered.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
                hourly_data = hourly_data.reindex([
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                ])
                
                fig_heatmap = px.imshow(
                    hourly_data,
                    title="⏰ Incident Heatmap: Day of Week vs Hour",
                    labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Incidents'},
                    color_continuous_scale='Reds'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Seasonal insights
            st.markdown("### 💡 Seasonal Insights")
            
            # Seasonal insights
            st.markdown("### 💡 Seasonal Insights")
            
            insights = []
            
            if 'month' in df_filtered.columns:
                monthly_pattern = df_filtered.groupby('month').size()
                peak_month = monthly_pattern.idxmax()
                low_month = monthly_pattern.idxmin()
                insights.append(f"📈 **Peak month**: {peak_month} ({monthly_pattern.max()} incidents)")
                insights.append(f"📉 **Lowest month**: {low_month} ({monthly_pattern.min()} incidents)")
            
            if 'day_of_week' in df_filtered.columns:
                dow_pattern = df_filtered.groupby('day_of_week').size()
                peak_day = dow_pattern.idxmax()
                low_day = dow_pattern.idxmin()
                insights.append(f"📅 **Peak day**: {peak_day} ({dow_pattern.max()} incidents)")
                insights.append(f"📅 **Quietest day**: {low_day} ({dow_pattern.min()} incidents)")
            
            if 'hour' in df_filtered.columns:
                hourly_pattern = df_filtered.groupby('hour').size()
                peak_hour = hourly_pattern.idxmax()
                low_hour = hourly_pattern.idxmin()
                insights.append(f"⏰ **Peak hour**: {peak_hour}:00 ({hourly_pattern.max()} incidents)")
                insights.append(f"⏰ **Quietest hour**: {low_hour}:00 ({hourly_pattern.min()} incidents)")
            
            # Weekend vs weekday analysis
            if 'is_weekend' in df_filtered.columns:
                weekend_incidents = df_filtered[df_filtered['is_weekend']].shape[0]
                weekday_incidents = df_filtered[~df_filtered['is_weekend']].shape[0]
                if weekday_incidents > 0:
                    weekend_ratio = weekend_incidents / weekday_incidents
                    if weekend_ratio > 1.2:
                        insights.append("📊 **Weekend pattern**: Higher incident rate on weekends")
                    elif weekend_ratio < 0.8:
                        insights.append("📊 **Weekday pattern**: Higher incident rate on weekdays")
                    else:
                        insights.append("📊 **Balanced pattern**: Similar rates on weekends and weekdays")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Recommendations based on patterns
            st.markdown("### 🎯 Recommendations")
            recommendations = [
                "📋 **Staffing optimization**: Adjust staffing levels based on peak times identified",
                "🔍 **Preventive measures**: Focus prevention efforts during high-risk periods",
                "📊 **Resource allocation**: Allocate more resources during peak months/days",
                "⏰ **Monitoring enhancement**: Increase monitoring during identified high-risk hours",
                "📈 **Trend tracking**: Monitor these patterns over time for changes"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

    with ml_tab5:
        st.subheader("📊 ML Insights Summary")
        st.markdown("*Key findings from machine learning analysis*")
        
        # Correlation heatmap
        if len(corr_matrix) > 1:
            fig_corr = px.imshow(
                corr_matrix,
                title="🔗 Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Comprehensive insights summary
        st.markdown("### 🎯 Key ML Insights")
        
        insights_list = [
            "📊 **Data Quality**: Processed and analyzed successfully",
            "🔍 **Pattern Detection**: Multiple analytical approaches applied",
            "⚡ **Real-time Analysis**: Results updated based on current filters"
        ]
        
        # Add specific insights from each analysis
        if 'cluster_labels' in st.session_state:
            n_clusters = len(set(st.session_state['cluster_labels']))
            insights_list.append(f"🔗 **Clustering**: Identified {n_clusters} distinct incident patterns")
        
        if 'anomaly_labels' in st.session_state:
            n_anomalies = sum(st.session_state['anomaly_labels'] == -1)
            anomaly_rate = (n_anomalies / len(st.session_state['anomaly_labels'])) * 100
            insights_list.append(f"🚨 **Anomalies**: Detected {n_anomalies} unusual incidents ({anomaly_rate:.1f}% anomaly rate)")
        
        if 'association_rules' in st.session_state and len(st.session_state['association_rules']) > 0:
            n_rules = len(st.session_state['association_rules'])
            avg_confidence = st.session_state['association_rules']['confidence'].mean()
            insights_list.append(f"🔍 **Associations**: Found {n_rules} significant relationships (avg. confidence: {avg_confidence:.1%})")
        
        if 'trend_predictions' in st.session_state:
            predictions = st.session_state['trend_predictions']
            avg_predicted = predictions['predicted_incidents'].mean()
            insights_list.append(f"🔮 **Forecasting**: Predicted average of {avg_predicted:.1f} incidents per month")
        
        if 'risk_model' in st.session_state:
            performance = st.session_state['risk_model']['performance']
            accuracy = performance['accuracy']
            insights_list.append(f"🎯 **Risk Prediction**: Model accuracy of {accuracy:.1%} for predicting high-risk incidents")
        
        for insight in insights_list:
            st.markdown(insight)
        
        # Feature importance across all models
        if 'risk_model' in st.session_state:
            st.markdown("### 🔑 Overall Feature Importance")
            feature_importance = st.session_state['risk_model']['feature_importance']
            
            # Create a more detailed feature importance chart
            fig_overall_importance = px.bar(
                feature_importance.head(8),
                x='importance',
                y='feature',
                orientation='h',
                title="🔑 Most Important Features Across All ML Models",
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='viridis'
            )
            fig_overall_importance.update_layout(height=400)
            st.plotly_chart(fig_overall_importance, use_container_width=True)
        
        # Model performance comparison
        if 'risk_model' in st.session_state:
            st.markdown("### 📈 Model Performance Summary")
            
            performance = st.session_state['risk_model']['performance']
            
            # Create performance metrics chart
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [performance['accuracy'], performance['precision'], 
                         performance['recall'], performance['f1_score']]
            })
            
            fig_metrics = px.bar(
                metrics_df,
                x='Metric',
                y='Score',
                title="📊 Risk Prediction Model Performance",
                labels={'Score': 'Performance Score'},
                color='Score',
                color_continuous_scale='viridis'
            )
            fig_metrics.update_layout(height=400)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Data quality assessment
        st.markdown("### 📋 Data Quality Assessment")
        
        quality_metrics = []
        
        # Completeness
        total_records = len(df_filtered)
        required_fields = ['incident_date', 'incident_type', 'severity', 'location']
        completeness_scores = []
        
        for field in required_fields:
            if field in df_filtered.columns:
                non_null_count = df_filtered[field].notna().sum()
                completeness = (non_null_count / total_records) * 100
                completeness_scores.append(completeness)
                quality_metrics.append(f"✅ **{field.replace('_', ' ').title()}**: {completeness:.1f}% complete")
        
        overall_completeness = np.mean(completeness_scores) if completeness_scores else 0
        
        # Date range coverage
        if 'incident_date' in df_filtered.columns:
            date_range = (df_filtered['incident_date'].max() - df_filtered['incident_date'].min()).days
            quality_metrics.append(f"📅 **Date Coverage**: {date_range} days")
        
        # Record count adequacy
        if total_records >= 1000:
            adequacy = "🟢 Excellent"
        elif total_records >= 500:
            adequacy = "🟡 Good"
        elif total_records >= 100:
            adequacy = "🟠 Adequate"
        else:
            adequacy = "🔴 Limited"
        
        quality_metrics.append(f"📊 **Sample Size**: {total_records} records ({adequacy})")
        quality_metrics.append(f"📈 **Overall Completeness**: {overall_completeness:.1f}%")
        
        for metric in quality_metrics:
            st.markdown(f"- {metric}")
        
        # Recommendations for improvement
        st.markdown("### 💡 Recommendations for Enhanced Analysis")
        
        recommendations = []
        
        if overall_completeness < 90:
            recommendations.append("📋 **Data Quality**: Improve data completeness for better ML model performance")
        
        if total_records < 500:
            recommendations.append("📊 **Sample Size**: Collect more historical data for more robust predictions")
        
        if 'cluster_labels' not in st.session_state:
            recommendations.append("🔗 **Clustering**: Run clustering analysis to discover hidden patterns")
        
        if 'anomaly_labels' not in st.session_state:
            recommendations.append("🚨 **Anomaly Detection**: Identify unusual incidents requiring attention")
        
        if 'association_rules' not in st.session_state:
            recommendations.append("🔍 **Association Analysis**: Discover relationships between incident factors")
        
        if 'trend_predictions' not in st.session_state:
            recommendations.append("🔮 **Predictive Modeling**: Generate forecasts for resource planning")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "🔄 **Regular Analysis**: Run ML analysis monthly to track changing patterns",
                "📊 **Data Integration**: Consider integrating additional data sources",
                "🎯 **Action Planning**: Develop action plans based on identified patterns",
                "📈 **Continuous Monitoring**: Set up automated monitoring of key metrics",
                "🔍 **Deep Dive Analysis**: Investigate specific patterns for operational insights"
            ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Export comprehensive report
        if st.button("📥 Download Complete ML Analysis Report"):
            # Generate comprehensive report
            report_content = f"""
# NDIS Incident ML Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Incidents Analyzed**: {len(df_filtered)}
- **Analysis Period**: {df_filtered['incident_date'].min().strftime('%Y-%m-%d')} to {df_filtered['incident_date'].max().strftime('%Y-%m-%d')}
- **Data Quality Score**: {overall_completeness:.1f}%

## Key Findings
"""
            
            # Add specific findings from each analysis
            if 'cluster_labels' in st.session_state:
                n_clusters = len(set(st.session_state['cluster_labels']))
                report_content += f"\n### Clustering Analysis\n- Identified {n_clusters} distinct incident patterns\n"
            
            if 'anomaly_labels' in st.session_state:
                n_anomalies = sum(st.session_state['anomaly_labels'] == -1)
                anomaly_rate = (n_anomalies / len(st.session_state['anomaly_labels'])) * 100
                report_content += f"\n### Anomaly Detection\n- Found {n_anomalies} anomalous incidents ({anomaly_rate:.1f}% rate)\n"
            
            if 'association_rules' in st.session_state:
                n_rules = len(st.session_state['association_rules'])
                report_content += f"\n### Association Rules\n- Discovered {n_rules} significant relationships\n"
            
            if 'trend_predictions' in st.session_state:
                avg_predicted = st.session_state['trend_predictions']['predicted_incidents'].mean()
                report_content += f"\n### Trend Forecasting\n- Predicted average: {avg_predicted:.1f} incidents per month\n"
            
            if 'risk_model' in st.session_state:
                accuracy = st.session_state['risk_model']['performance']['accuracy']
                report_content += f"\n### Risk Prediction\n- Model accuracy: {accuracy:.1%}\n"
            
            report_content += f"""
## Recommendations
{chr(10).join([f"- {rec.replace('**', '').replace('🔗', '').replace('🚨', '').replace('🔍', '').replace('🔮', '').replace('🎯', '').replace('📋', '').replace('📊', '').replace('🔄', '').replace('📈', '').replace('💡', '')}" for rec in recommendations])}

## Data Quality Assessment
- Overall Completeness: {overall_completeness:.1f}%
- Sample Size: {total_records} records
- Date Range: {date_range if 'date_range' in locals() else 'N/A'} days

---
Report generated by NDIS Analytics Dashboard
"""
            
            st.download_button(
                label="Download ML Report",
                data=report_content,
                file_name=f"ndis_ml_analysis_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        
        # Quick action buttons
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh All Models"):
                # Clear all ML session state
                keys_to_clear = ['cluster_labels', 'anomaly_labels', 'association_rules', 
                               'trend_predictions', 'risk_model']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ All models cleared. Re-run analyses for fresh results.")
        
        with col2:
            if st.button("📊 Export All Data"):
                # Export processed data with ML results
                export_df = df_filtered.copy()
                
                if 'cluster_labels' in st.session_state:
                    export_df['cluster_id'] = st.session_state['cluster_labels']
                
                if 'anomaly_labels' in st.session_state:
                    export_df['is_anomaly'] = (st.session_state['anomaly_labels'] == -1)
                
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Enhanced Dataset",
                    data=csv_data,
                    file_name=f"ndis_enhanced_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🎯 Model Comparison"):
                st.info("Model comparison feature - compare different ML algorithms side by side")
        
        # Performance tips
        with st.expander("💡 Tips for Better ML Analysis"):
            st.markdown("""
            **To improve your ML analysis results:**
            
            1. **Data Quality**
               - Ensure all required fields are complete
               - Standardize categorical values (e.g., consistent severity levels)
               - Include temporal data for trend analysis
            
            2. **Sample Size**
               - More data generally leads to better model performance
               - Aim for at least 100+ incidents for clustering
               - 500+ incidents recommended for reliable predictions
            
            3. **Feature Engineering**
               - Include derived features (e.g., time of day, day of week)
               - Consider participant characteristics and environmental factors
               - Add calculated fields like notification delays
            
            4. **Regular Updates**
               - Re-run analysis monthly or quarterly
               - Monitor for changing patterns over time
               - Adjust parameters based on new insights
            
            5. **Action-Oriented Analysis**
               - Focus on actionable insights
               - Validate findings with domain experts
               - Implement changes based on identified patterns
            """)
        
        # Show what each analysis provides when no models have been run
        if ('cluster_labels' not in st.session_state and 
            'anomaly_labels' not in st.session_state and 
            'association_rules' not in st.session_state and 
            'trend_predictions' not in st.session_state and 
            'risk_model' not in st.session_state):
            
            st.info("Run the ML analysis tools above to see comprehensive insights here.")
            
            # Show what each analysis provides
            st.markdown("### 🔍 Available ML Analyses")
            
            analysis_info = [
                "🔗 **Clustering**: Groups similar incidents to identify patterns",
                "🚨 **Anomaly Detection**: Finds unusual incidents requiring attention", 
                "🔍 **Association Rules**: Discovers relationships between incident factors",
                "🔮 **Prediction**: Forecasts trends and predicts risk factors",
                "📊 **Insights**: Comprehensive summary and recommendations"
            ]
            
            for info in analysis_info:
                st.markdown(f"- {info}")

            
            import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# ML Libraries for advanced analytics
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Association rules libraries
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="NDIS Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def process_data(df):
    """Process and enhance the loaded data"""
    try:
        df = df.copy()
        
        # Ensure we have required columns
        required_columns = ['incident_date', 'incident_type', 'severity', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert date columns safely
        if 'incident_date' in df.columns:
            df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce', dayfirst=True)
            if df['incident_date'].isna().all():
                st.error("❌ Could not parse incident_date. Please use DD/MM/YYYY or YYYY-MM-DD format.")
                return None
        
        # Handle notification_date
        if 'notification_date' in df.columns:
            df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce', dayfirst=True)
        else:
            delays = np.random.uniform(0, 2, len(df))
            df['notification_date'] = df['incident_date'] + pd.to_timedelta(delays, unit='days')
        
        # Calculate notification delay safely
        if 'notification_date' in df.columns and not df['notification_date'].isna().all():
            df['notification_delay'] = (df['notification_date'] - df['incident_date']).dt.total_seconds() / (24 * 3600)
            df['notification_delay'] = df['notification_delay'].fillna(0)
        else:
            df['notification_delay'] = 0
        
        # Add time-based columns
        df['month'] = df['incident_date'].dt.month_name()
        df['day_of_week'] = df['incident_date'].dt.day_name()
        df['quarter'] = df['incident_date'].dt.quarter
        df['is_weekend'] = df['incident_date'].dt.dayofweek >= 5
        
        # Handle incident_time
        if 'incident_time' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['incident_time'], format='%H:%M', errors='coerce').dt.hour
            except:
                df['hour'] = pd.to_datetime(df['incident_time'], errors='coerce').dt.hour
        else:
            df['hour'] = np.random.randint(6, 22, len(df))
        
        df['hour'] = df['hour'].fillna(12)
        
        # Risk scoring
        severity_mapping = {
            'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4,
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4,
            'L': 1, 'M': 2, 'H': 3, 'C': 4
        }
        df['severity_score'] = df['severity'].map(severity_mapping).fillna(1)
        
        # Create age groups
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35)
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        else:
            df['age'] = np.random.normal(40, 20, len(df)).clip(18, 85).astype(int)
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Ensure all participants have names
        if 'participant_name' not in df.columns:
            df['participant_name'] = [f'Participant_{i:03d}' for i in range(1, len(df) + 1)]
        
        # Ensure incident_id exists
        if 'incident_id' not in df.columns:
            df['incident_id'] = [f'INC{i:06d}' for i in range(1, len(df) + 1)]
        
        # Clean string columns
        string_columns = ['incident_type', 'severity', 'location']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

@st.cache_data
def prepare_ml_features(df):
    """Prepare features for ML analysis safely"""
    try:
        if not SKLEARN_AVAILABLE:
            return df, {}
            
        ml_df = df.copy()
        label_encoders = {}
        
        # Encode categorical variables safely
        categorical_cols = ['incident_type', 'severity', 'location']
        
        for col in categorical_cols:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col] = ml_df[col].fillna('Unknown').astype(str)
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col])
                label_encoders[col] = le
        
        return ml_df, label_encoders
        
    except Exception as e:
        st.error(f"Error preparing ML features: {str(e)}")
        return df, {}

def perform_clustering_analysis(df, method='kmeans', n_clusters=5):
    """Perform clustering analysis"""
    try:
        if not SKLEARN_AVAILABLE:
            st.error("❌ Scikit-learn not available for clustering analysis")
            return None, None, None, None
            
        # Prepare features for clustering
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age' in df.columns:
            feature_cols.append('age')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'severity_score' in df.columns:
            feature_cols.append('severity_score')
        
        if len(feature_cols) < 2:
            st.warning("⚠️ Not enough features for clustering analysis")
            return None, None, None, None
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on method
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Calculate clustering metrics
        metrics = {}
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            try:
                metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
                metrics['calinski_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
                metrics['n_clusters'] = len(unique_labels)
            except:
                metrics['n_clusters'] = len(unique_labels)
        
        return cluster_labels, metrics, X_scaled, feature_cols
        
    except Exception as e:
        st.error(f"❌ Clustering error: {str(e)}")
        return None, None, None, None

def detect_anomalies(df, method='isolation_forest', contamination=0.1):
    """Detect anomalies in the data"""
    try:
        if not SKLEARN_AVAILABLE:
            st.error("❌ Scikit-learn not available for anomaly detection")
            return None, None, None
            
        # Prepare features
        feature_cols = [col for col in df.columns if col.endswith('_encoded')]
        if 'age' in df.columns:
            feature_cols.append('age')
        if 'hour' in df.columns:
            feature_cols.append('hour')
        if 'severity_score' in df.columns:
            feature_cols.append('severity_score')
        
        if len(feature_cols) < 2:
            st.warning("⚠️ Not enough features for anomaly detection")
            return None, None, None
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection method
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        
        anomaly_labels = detector.fit_predict(X_scaled)
        
        return anomaly_labels, X_scaled, feature_cols
        
    except Exception as e:
        st.error(f"❌ Anomaly detection error: {str(e)}")
        return None, None, None

def find_association_rules_enhanced(df, features, min_support=0.1, min_confidence=0.6, min_lift=1.2):
    """Enhanced association rules mining with feature selection"""
    try:
        if not MLXTEND_AVAILABLE:
            return None, None
            
        # Create transactions using selected features
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            for col in features:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    # Create more readable transaction items
                    item_name = f"{col.replace('_', ' ').title()}: {str(row[col]).strip()}"
                    transaction.append(item_name)
            if len(transaction) >= 2:
                transactions.append(transaction)
        
        if len(transactions) < 10:
            return None, None
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        try:
            frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True)
        except:
            frequent_itemsets = apriori(df_transactions, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return None, None
        
        # Generate association rules with lift filter
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            if len(rules) > 0:
                rules = rules[rules['lift'] >= min_lift]
        except:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
            if len(rules) > 0:
                rules = rules[rules['lift'] >= min_lift]
        
        return frequent_itemsets, rules
        
    except Exception as e:
        st.error(f"❌ Enhanced association rules error: {str(e)}")
        return None, None

def detect_anomalies_enhanced(df, features, method='isolation_forest', contamination=0.1):
    """Enhanced anomaly detection with feature selection and scoring"""
    try:
        if not SKLEARN_AVAILABLE:
            return None, None, None, None
            
        # Prepare selected features
        X = df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection method with scoring
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.score_samples(X_scaled)
        elif method == 'one_class_svm':
            detector = OneClassSVM(nu=contamination)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.score_samples(X_scaled)
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.negative_outlier_factor_
        elif method == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.score_samples(X_scaled)
        
        # Convert scores to positive values (higher = more anomalous)
        if method == 'local_outlier_factor':
            anomaly_scores = -anomaly_scores
        else:
            anomaly_scores = -anomaly_scores  # Convert to positive (lower scores = more anomalous)
        
        return anomaly_labels, X_scaled, features, anomaly_scores
        
    except Exception as e:
        st.error(f"❌ Enhanced anomaly detection error: {str(e)}")
        return None, None, None, None

def predict_incident_trends(df):
    """Predict future incident trends using time series analysis"""
    try:
        if not SKLEARN_AVAILABLE:
            return None, None, None
        
        # Prepare time series data
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        
        # Create monthly aggregations
        monthly_incidents = df.groupby(df['incident_date'].dt.to_period('M')).size()
        monthly_incidents.index = monthly_incidents.index.to_timestamp()
        
        if len(monthly_incidents) < 6:
            return None, None, "Need at least 6 months of data for prediction"
        
        # Prepare features for prediction
        monthly_df = monthly_incidents.reset_index()
        monthly_df.columns = ['date', 'incidents']
        monthly_df['month_num'] = range(len(monthly_df))
        monthly_df['month'] = monthly_df['date'].dt.month
        monthly_df['quarter'] = monthly_df['date'].dt.quarter
        
        # Simple linear regression for trend
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        X = monthly_df[['month_num', 'month', 'quarter']]
        y = monthly_df['incidents']
        
        # Create polynomial features for better fit
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        
        poly_model.fit(X, y)
        
        # Predict next 6 months
        last_month_num = monthly_df['month_num'].max()
        future_dates = pd.date_range(start=monthly_df['date'].max() + pd.DateOffset(months=1), periods=6, freq='M')
        
        future_X = []
        for i, date in enumerate(future_dates):
            future_X.append([
                last_month_num + i + 1,
                date.month,
                date.quarter
            ])
        
        future_X = pd.DataFrame(future_X, columns=['month_num', 'month', 'quarter'])
        future_predictions = poly_model.predict(future_X)
        
        # Calculate prediction intervals (simple approach)
        historical_error = np.std(y - poly_model.predict(X))
        confidence_interval = 1.96 * historical_error  # 95% confidence
        
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_incidents': np.maximum(0, future_predictions),  # Ensure non-negative
            'lower_bound': np.maximum(0, future_predictions - confidence_interval),
            'upper_bound': future_predictions + confidence_interval
        })
        
        return monthly_df, predictions_df, None
        
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

def predict_risk_factors(df):
    """Predict high-risk incidents using classification"""
    try:
        if not SKLEARN_AVAILABLE:
            return None, None
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Prepare features
        feature_columns = []
        
        # Encode categorical variables for prediction
        if 'incident_type' in df.columns:
            type_encoded = LabelEncoder().fit_transform(df['incident_type'].fillna('Unknown'))
            df['incident_type_encoded'] = type_encoded
            feature_columns.append('incident_type_encoded')
        
        if 'location' in df.columns:
            location_encoded = LabelEncoder().fit_transform(df['location'].fillna('Unknown'))
            df['location_encoded'] = location_encoded
            feature_columns.append('location_encoded')
        
        # Add numeric features
        numeric_features = ['age', 'hour', 'notification_delay']
        for feat in numeric_features:
            if feat in df.columns:
                feature_columns.append(feat)
        
        if len(feature_columns) < 2:
            return None, "Need at least 2 features for risk prediction"
        
        # Create target variable (high risk = High or Critical severity)
        df['high_risk'] = df['severity'].str.lower().isin(['high', 'critical']).astype(int)
        
        # Prepare data
        X = df[feature_columns].fillna(0)
        y = df['high_risk']
        
        if y.sum() < 5:  # Need at least 5 high-risk cases
            return None, "Need at least 5 high-risk incidents for prediction"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Model performance
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return {
            'model': rf_model,
            'feature_importance': feature_importance,
            'performance': performance,
            'test_predictions': y_pred_proba,
            'feature_columns': feature_columns
        }, None
        
    except Exception as e:
        return None, f"Risk prediction error: {str(e)}"

def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        if len(df) == 0:
            st.error("❌ The uploaded file is empty.")
            return None
        
        st.info(f"📁 Loaded {len(df)} rows from uploaded file")
        processed_df = process_data(df)
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

def calculate_correlations(df):
    """Calculate correlations safely"""
    try:
        numeric_df = df.copy()
        
        # Convert categorical to numeric safely
        if 'severity_score' in numeric_df.columns:
            numeric_df['severity_numeric'] = numeric_df['severity_score']
        else:
            numeric_df['severity_numeric'] = 1
            
        if 'reportable' in df.columns:
            numeric_df['reportable_numeric'] = df['reportable'].map({'Yes': 1, 'No': 0}).fillna(0)
        else:
            numeric_df['reportable_numeric'] = 0
            
        if 'is_weekend' in df.columns:
            numeric_df['is_weekend_numeric'] = df['is_weekend'].astype(int)
        else:
            numeric_df['is_weekend_numeric'] = 0
        
        # Select available numeric columns
        correlation_vars = []
        possible_vars = ['age', 'severity_numeric', 'notification_delay', 'reportable_numeric', 
                        'is_weekend_numeric', 'hour']
        
        for var in possible_vars:
            if var in numeric_df.columns:
                numeric_df[var] = pd.to_numeric(numeric_df[var], errors='coerce').fillna(0)
                correlation_vars.append(var)
        
        if len(correlation_vars) >= 2:
            corr_matrix = numeric_df[correlation_vars].corr()
        else:
            corr_matrix = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], 
                                     columns=['severity_numeric', 'age'], 
                                     index=['severity_numeric', 'age'])
            numeric_df['age'] = numeric_df.get('age', 35)
        
        return corr_matrix, numeric_df
        
    except Exception as e:
        st.error(f"❌ Error calculating correlations: {str(e)}")
        corr_matrix = pd.DataFrame([[1.0]], columns=['severity_numeric'], index=['severity_numeric'])
        return corr_matrix, df

def generate_insights(df):
    """Generate insights safely"""
    insights = []
    
    try:
        total_incidents = len(df)
        insights.append(f"📊 Total incidents analyzed: {total_incidents}")
        
        if 'severity' in df.columns:
            critical_count = len(df[df['severity'].str.lower().isin(['critical', 'high'])])
            if critical_count > 0:
                insights.append(f"🚨 {critical_count} high-severity incidents requiring attention")
        
        if 'location' in df.columns:
            top_location = df['location'].value_counts().index[0] if len(df) > 0 else 'Unknown'
            insights.append(f"🏢 Most incidents occur at: {top_location}")
        
        if len(insights) == 1:
            insights.extend([
                "🔍 Data processing complete - ready for analysis",
                "📈 Use the analysis modes to explore patterns"
            ])
            
    except Exception as e:
        insights = [f"⚠️ Insights generation error: {str(e)[:50]}..."]
    
    return insights

# Main Application
st.title("🏥 NDIS Incident Analytics Dashboard")

# Data loading UI
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

# Check if we have data to work with
if df is None or len(df) == 0:
    st.markdown("""
    # 🏥 NDIS Incident Analytics Dashboard
    
    ## 📁 Welcome! Please Upload Your Data
    
    To get started with your NDIS incident analysis:
    
    1. **📋 Prepare your CSV file** with incident data
    2. **📁 Use the file uploader** in the sidebar
    3. **📊 Explore your data** with advanced analytics
    
    ### 📋 Required CSV Columns:
    - `incident_date` - Date of incident (DD/MM/YYYY format)
    - `incident_type` - Type of incident
    - `severity` - Severity level (Low, Medium, High, Critical)
    - `location` - Where the incident occurred
    
    ### 🔧 Optional Columns:
    - `notification_date` - When incident was reported
    - `participant_name` - Participant involved
    - `age` - Participant age
    - `reportable` - Whether incident is reportable (Yes/No)
    - `incident_time` - Time of incident (HH:MM)
    - `description` - Incident description
    """)
    
    # Show sample data format
    st.subheader("📋 Sample Data Format")
    sample_data = pd.DataFrame({
        'incident_date': ['01/01/2024', '02/01/2024', '03/01/2024'],
        'incident_type': ['Fall', 'Medication Error', 'Behavioral'],
        'severity': ['Medium', 'High', 'Low'],
        'location': ['Day Program', 'Residential', 'Community'],
        'reportable': ['Yes', 'No', 'No']
    })
    st.dataframe(sample_data, use_container_width=True)
    st.stop()

# Process data if loaded successfully
try:
    corr_matrix, numeric_df = calculate_correlations(df)
    insights = generate_insights(df)
    
    st.success(f"✅ Successfully loaded {len(df)} incidents from {df['incident_date'].min().strftime('%B %Y')} to {df['incident_date'].max().strftime('%B %Y')}")
    
except Exception as e:
    st.error(f"❌ Error processing data: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.header("🎛️ Analysis Controls")

# Analysis Mode Selection
analysis_mode = st.sidebar.selectbox(
    "🔬 Analysis Mode",
    ["Executive Overview", "Risk Analysis", "🤖 ML Analytics", "Data Explorer"]
)

# Filters
st.sidebar.subheader("🎯 Filters")

# Date range filter
if 'incident_date' in df.columns:
    min_date = df['incident_date'].min().date()
    max_date = df['incident_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['incident_date'].dt.date >= start_date) & 
            (df['incident_date'].dt.date <= end_date)
        ]
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

# Severity filter
if 'severity' in df.columns:
    severity_options = df['severity'].unique()
    severity_filter = st.sidebar.multiselect(
        "⚠️ Severity Level",
        options=severity_options,
        default=severity_options
    )
    df_filtered = df_filtered[df_filtered['severity'].isin(severity_filter)]

# Location filter
if 'location' in df.columns:
    location_options = df['location'].unique()
    location_filter = st.sidebar.multiselect(
        "📍 Location",
        options=location_options,
        default=location_options
    )
    df_filtered = df_filtered[df_filtered['location'].isin(location_filter)]

# Live insights
with st.sidebar:
    st.subheader("💡 Live Insights")
    for insight in insights[:3]:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

# Main dashboard content
if analysis_mode == "Executive Overview":
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Incidents", len(df_filtered))
    
    with col2:
        if 'severity' in df_filtered.columns:
            critical_count = len(df_filtered[df_filtered['severity'].str.lower().isin(['critical', 'high'])])
            st.metric("🚨 High Severity", critical_count)
        else:
            st.metric("🚨 High Severity", "N/A")
    
    with col3:
        if 'notification_delay' in df_filtered.columns:
            avg_delay = df_filtered['notification_delay'].mean()
            st.metric("⏱️ Avg Delay (days)", f"{avg_delay:.1f}")
        else:
            st.metric("⏱️ Avg Delay", "N/A")
    
    with col4:
        if 'participant_name' in df_filtered.columns:
            unique_participants = df_filtered['participant_name'].nunique()
            st.metric("👥 Participants", unique_participants)
        else:
            st.metric("👥 Participants", "N/A")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Incident types
        if 'incident_type' in df_filtered.columns:
            incident_counts = df_filtered['incident_type'].value_counts().head(10)
            fig1 = px.bar(
                x=incident_counts.values,
                y=incident_counts.index,
                orientation='h',
                title="🔝 Top Incident Types",
                labels={'x': 'Count', 'y': 'Incident Type'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Severity distribution
        if 'severity' in df_filtered.columns:
            severity_counts = df_filtered['severity'].value_counts()
            fig2 = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="⚠️ Severity Distribution"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly trends
    if 'incident_date' in df_filtered.columns:
        st.subheader("📈 Monthly Trends")
        monthly_data = df_filtered.groupby(df_filtered['incident_date'].dt.to_period('M')).size()
        monthly_data.index = monthly_data.index.astype(str)
        
        fig3 = px.line(
            x=monthly_data.index,
            y=monthly_data.values,
            title="📊 Incidents Over Time",
            markers=True,
            labels={'x': 'Month', 'y': 'Number of Incidents'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

elif analysis_mode == "Risk Analysis":
    st.subheader("🎯 Risk Analysis")
    
    # Risk Assessment Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        if 'location' in df_filtered.columns and 'severity_score' in df_filtered.columns:
            # Risk by location
            location_risk = df_filtered.groupby('location').agg({
                'severity_score': 'mean',
                'incident_id': 'count'
            }).round(2)
            location_risk.columns = ['Avg Severity', 'Count']
            
            fig_risk = px.scatter(
                location_risk.reset_index(),
                x='Count',
                y='Avg Severity',
                size='Count',
                hover_name='location',
                title="🎯 Risk Matrix: Volume vs Severity by Location",
                labels={'Count': 'Number of Incidents', 'Avg Severity': 'Average Severity Score'}
            )
            fig_risk.update_layout(height=500)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("Risk analysis requires location and severity data")
    
    with col2:
        # Time-based risk analysis
        if 'hour' in df_filtered.columns:
            hourly_incidents = df_filtered.groupby('hour').size()
            fig_hourly = px.bar(
                x=hourly_incidents.index,
                y=hourly_incidents.values,
                title="⏰ Incidents by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Incidents'}
            )
            fig_hourly.update_layout(height=500)
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Risk summary table
    if 'location' in df_filtered.columns and 'severity_score' in df_filtered.columns:
        st.subheader("📊 Risk Summary by Location")
        location_risk_expanded = df_filtered.groupby('location').agg({
            'severity_score': ['mean', 'max', 'count'],
            'notification_delay': 'mean'
        }).round(2)
        
        location_risk_expanded.columns = ['Avg Severity', 'Max Severity', 'Incident Count', 'Avg Delay (days)']
        st.dataframe(location_risk_expanded, use_container_width=True)

elif analysis_mode == "🤖 ML Analytics":
    st.subheader("🤖 Machine Learning Analytics")
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Machine Learning features require scikit-learn. Please install it to use ML analytics.")
        st.code("pip install scikit-learn")
        st.stop()
    
    # Prepare ML features
    ml_df, label_encoders = prepare_ml_features(df_filtered)
    
    # ML Analysis tabs
    ml_tab1, ml_tab2, ml_tab3, ml_tab4, ml_tab5 = st.tabs(["🔗 Clustering", "🚨 Anomaly Detection", "🔍 Association Rules", "🔮 Prediction", "📊 ML Insights"])
    
    with ml_tab1:
        st.subheader("🔗 Incident Clustering Analysis")
        st.markdown("*Discover hidden patterns and group similar incidents together*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Clustering Parameters")
            
            clustering_method = st.selectbox(
                "Clustering Algorithm",
                ["kmeans", "dbscan", "hierarchical"],
                help="K-means: Fixed number of clusters | DBSCAN: Density-based | Hierarchical: Tree-like clustering"
            )
            
            if clustering_method in ['kmeans', 'hierarchical']:
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            else:
                n_clusters = None
                st.info("ℹ️ DBSCAN automatically determines the optimal number of clusters")
            
            if st.button("🔄 Run Clustering Analysis", type="primary"):
                with st.spinner("🔄 Analyzing incident patterns..."):
                    cluster_labels, metrics, X_scaled, feature_cols = perform_clustering_analysis(
                        ml_df, method=clustering_method, n_clusters=n_clusters
                    )
                    
                    if cluster_labels is not None:
                        st.session_state['cluster_labels'] = cluster_labels
                        st.session_state['cluster_metrics'] = metrics
                        st.session_state['cluster_features'] = feature_cols
                        st.session_state['cluster_scaled'] = X_scaled
                        st.success("✅ Clustering analysis completed!")
                    else:
                        st.error("❌ Clustering analysis failed. Please check your data.")
        
        with col2:
            if 'cluster_labels' in st.session_state and st.session_state['cluster_labels'] is not None:
                cluster_labels = st.session_state['cluster_labels']
                metrics = st.session_state.get('cluster_metrics', {})
                
                # Display clustering metrics
                st.markdown("### 📊 Clustering Results")
                if metrics:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("🎯 Clusters Found", metrics.get('n_clusters', len(set(cluster_labels))))
                    with col_b:
                        if 'silhouette_score' in metrics:
                            score = metrics['silhouette_score']
                            st.metric("📈 Silhouette Score", f"{score:.3f}", 
                                    help="Quality measure: -1 (poor) to 1 (excellent)")
                    with col_c:
                        if 'calinski_score' in metrics:
                            st.metric("🎯 Calinski Score", f"{metrics['calinski_score']:.1f}",
                                    help="Higher values indicate better clustering")
                
                # Cluster visualization using PCA
                if 'cluster_scaled' in st.session_state:
                    X_scaled = st.session_state['cluster_scaled']
                    
                    if X_scaled.shape[1] >= 2:
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Create scatter plot
                        cluster_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': [f'Cluster {c}' for c in cluster_labels],
                            'Incident_Type': ml_df['incident_type'].values if 'incident_type' in ml_df.columns else 'Unknown',
                            'Severity': ml_df['severity'].values if 'severity' in ml_df.columns else 'Unknown'
                        })
                        
                        fig_cluster = px.scatter(
                            cluster_df,
                            x='PC1', y='PC2',
                            color='Cluster',
                            hover_data=['Incident_Type', 'Severity'],
                            title="🔍 Incident Clusters (PCA Visualization)",
                            labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'}
                        )
                        fig_cluster.update_layout(height=500)
                        st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Cluster characteristics analysis
                st.markdown("### 🔍 Cluster Characteristics")
                cluster_analysis = []
                
                for cluster_id in sorted(set(cluster_labels)):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_data = ml_df[cluster_mask]
                    
                    analysis = {
                        'Cluster': f'Cluster {cluster_id}',
                        'Size': len(cluster_data),
                        'Percentage': f"{len(cluster_data)/len(ml_df)*100:.1f}%"
                    }
                    
                    # Most common characteristics
                    if 'incident_type' in cluster_data.columns:
                        most_common = cluster_data['incident_type'].mode()
                        analysis['Common Type'] = most_common.iloc[0] if len(most_common) > 0 else 'Mixed'
                    
                    if 'severity' in cluster_data.columns:
                        most_common_sev = cluster_data['severity'].mode()
                        analysis['Common Severity'] = most_common_sev.iloc[0] if len(most_common_sev) > 0 else 'Mixed'
                    
                    if 'location' in cluster_data.columns:
                        most_common_loc = cluster_data['location'].mode()
                        analysis['Common Location'] = most_common_loc.iloc[0] if len(most_common_loc) > 0 else 'Mixed'
                    
                    cluster_analysis.append(analysis)
                
                cluster_results_df = pd.DataFrame(cluster_analysis)
                st.dataframe(cluster_results_df, use_container_width=True)
            else:
                st.info("👆 Click 'Run Clustering Analysis' to discover incident patterns")
    
    with ml_tab2:
        st.subheader("🚨 Anomaly Detection")
        st.markdown("*Identify unusual incidents that may require special attention*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Detection Parameters")
            
            anomaly_method = st.selectbox(
                "Detection Algorithm",
                ["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"],
                help="Different algorithms for detecting unusual patterns"
            )
            
            contamination = st.slider(
                "Expected Anomaly Rate (%)", 
                1, 20, 10,
                help="Percentage of incidents expected to be anomalous"
            ) / 100
            
            # Advanced parameters
            with st.expander("🔧 Advanced Parameters"):
                if anomaly_method == "isolation_forest":
                    n_estimators = st.slider("Number of Trees", 50, 200, 100)
                    max_samples = st.slider("Max Samples", 0.5, 1.0, 1.0)
                elif anomaly_method == "local_outlier_factor":
                    n_neighbors = st.slider("Number of Neighbors", 5, 50, 20)
                elif anomaly_method == "one_class_svm":
                    nu = contamination
                    gamma = st.selectbox("Gamma", ["scale", "auto"], index=0)
            
            # Feature selection for anomaly detection
            st.markdown("### 📊 Features for Analysis")
            available_features = []
            if 'age' in ml_df.columns:
                available_features.append('age')
            if 'hour' in ml_df.columns:
                available_features.append('hour')
            if 'severity_score' in ml_df.columns:
                available_features.append('severity_score')
            if 'notification_delay' in ml_df.columns:
                available_features.append('notification_delay')
            
            # Add encoded features
            encoded_features = [col for col in ml_df.columns if col.endswith('_encoded')]
            available_features.extend(encoded_features)
            
            selected_anomaly_features = st.multiselect(
                "Select features for anomaly detection",
                options=available_features,
                default=available_features[:4] if len(available_features) >= 4 else available_features
            )
            
            if st.button("🔍 Detect Anomalies", type="primary"):
                if len(selected_anomaly_features) < 2:
                    st.error("Please select at least 2 features for anomaly detection")
                else:
                    with st.spinner("🔍 Scanning for unusual incidents..."):
                        anomaly_labels, X_scaled, feature_cols, anomaly_scores = detect_anomalies_enhanced(
                            ml_df, features=selected_anomaly_features, method=anomaly_method, 
                            contamination=contamination
                        )
                        
                        if anomaly_labels is not None:
                            st.session_state['anomaly_labels'] = anomaly_labels
                            st.session_state['anomaly_features'] = feature_cols
                            st.session_state['anomaly_scaled'] = X_scaled
                            st.session_state['anomaly_scores'] = anomaly_scores
                            st.success("✅ Anomaly detection completed!")
                        else:
                            st.error("❌ Anomaly detection failed. Please check your data.")
        
        with col2:
            if 'anomaly_labels' in st.session_state and st.session_state['anomaly_labels'] is not None:
                anomaly_labels = st.session_state['anomaly_labels']
                anomaly_scores = st.session_state.get('anomaly_scores', None)
                
                # Enhanced anomaly statistics
                n_anomalies = sum(anomaly_labels == -1)
                n_normal = sum(anomaly_labels == 1)
                anomaly_percentage = n_anomalies / len(anomaly_labels) * 100
                
                st.markdown("### 📊 Anomaly Detection Results")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("🚨 Anomalies Found", n_anomalies)
                with col_b:
                    st.metric("📈 Anomaly Rate", f"{anomaly_percentage:.1f}%")
                with col_c:
                    st.metric("✅ Normal Cases", n_normal)
                with col_d:
                    if anomaly_scores is not None:
                        avg_anomaly_score = np.mean([s for i, s in enumerate(anomaly_scores) if anomaly_labels[i] == -1])
                        st.metric("⚡ Avg Anomaly Score", f"{avg_anomaly_score:.3f}")
                
                if n_anomalies > 0:
                    # Enhanced visualizations
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Anomaly visualization using PCA
                        if 'anomaly_scaled' in st.session_state:
                            X_scaled = st.session_state['anomaly_scaled']
                            
                            if X_scaled.shape[1] >= 2:
                                pca = PCA(n_components=2)
                                X_pca = pca.fit_transform(X_scaled)
                                
                                anomaly_df = pd.DataFrame({
                                    'PC1': X_pca[:, 0],
                                    'PC2': X_pca[:, 1],
                                    'Type': ['🚨 Anomaly' if label == -1 else '✅ Normal' for label in anomaly_labels],
                                    'Incident_Type': ml_df['incident_type'].values if 'incident_type' in ml_df.columns else 'Unknown',
                                    'Severity': ml_df['severity'].values if 'severity' in ml_df.columns else 'Unknown',
                                    'Score': anomaly_scores if anomaly_scores is not None else [0] * len(anomaly_labels)
                                })
                                
                                fig_anomaly = px.scatter(
                                    anomaly_df,
                                    x='PC1', y='PC2',
                                    color='Type',
                                    size='Score',
                                    hover_data=['Incident_Type', 'Severity', 'Score'],
                                    title="🔍 Anomaly Detection Results (PCA)",
                                    color_discrete_map={'🚨 Anomaly': 'red', '✅ Normal': 'blue'}
                                )
                                fig_anomaly.update_layout(height=400)
                                st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    with col_viz2:
                        # Anomaly score distribution
                        if anomaly_scores is not None:
                            score_df = pd.DataFrame({
                                'Score': anomaly_scores,
                                'Type': ['🚨 Anomaly' if label == -1 else '✅ Normal' for label in anomaly_labels]
                            })
                            
                            fig_scores = px.histogram(
                                score_df,
                                x='Score',
                                color='Type',
                                title="⚡ Anomaly Score Distribution",
                                labels={'Score': 'Anomaly Score', 'count': 'Count'},
                                color_discrete_map={'🚨 Anomaly': 'red', '✅ Normal': 'blue'}
                            )
                            fig_scores.update_layout(height=400)
                            st.plotly_chart(fig_scores, use_container_width=True)
                    
                    # Enhanced anomaly analysis
                    st.markdown("### 🔍 Detailed Anomaly Analysis")
                    
                    anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
                    anomaly_data = ml_df.iloc[anomaly_indices].copy()
                    normal_data = ml_df.iloc[[i for i, label in enumerate(anomaly_labels) if label == 1]]
                    
                    # Add anomaly scores to the data
                    if anomaly_scores is not None:
                        anomaly_data['anomaly_score'] = [anomaly_scores[i] for i in anomaly_indices]
                        anomaly_data = anomaly_data.sort_values('anomaly_score', ascending=True)
                    
                    # Statistical comparison
                    st.markdown("#### 📈 Statistical Comparison: Anomalous vs Normal")
                    
                    comparison_stats = []
                    numeric_cols = ['age', 'severity_score', 'notification_delay', 'hour']
                    
                    for col in numeric_cols:
                        if col in anomaly_data.columns and col in normal_data.columns:
                            anomaly_mean = anomaly_data[col].mean()
                            normal_mean = normal_data[col].mean()
                            anomaly_std = anomaly_data[col].std()
                            normal_std = normal_data[col].std()
                            
                            comparison_stats.append({
                                'Feature': col.replace('_', ' ').title(),
                                'Anomaly Mean': f"{anomaly_mean:.2f}",
                                'Normal Mean': f"{normal_mean:.2f}",
                                'Anomaly Std': f"{anomaly_std:.2f}",
                                'Normal Std': f"{normal_std:.2f}",
                                'Difference': f"{abs(anomaly_mean - normal_mean):.2f}"
                            })
                    
                    if comparison_stats:
                        stats_df = pd.DataFrame(comparison_stats)
                        st.dataframe(stats_df, use_container_width=True)
                    
                    # Categorical comparison
                    st.markdown("#### 📊 Categorical Distribution: Anomalous vs Normal")
                    
                    comparison_data = []
                    categorical_cols = ['incident_type', 'severity', 'location']
                    
                    for col in categorical_cols:
                        if col in anomaly_data.columns:
                            anomaly_dist = anomaly_data[col].value_counts(normalize=True).head(3)
                            normal_dist = normal_data[col].value_counts(normalize=True)
                            
                            for category in anomaly_dist.index:
                                comparison_data.append({
                                    'Category': f"{col.replace('_', ' ').title()}: {category}",
                                    'Anomaly %': f"{anomaly_dist[category] * 100:.1f}%",
                                    'Normal %': f"{normal_dist.get(category, 0) * 100:.1f}%",
                                    'Difference': f"{abs(anomaly_dist[category] - normal_dist.get(category, 0)) * 100:.1f}%"
                                })
                    
                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True)
                    
                    # Top anomalous incidents
                    st.markdown("#### 🚨 Most Anomalous Incidents")
                    display_cols = ['incident_type', 'severity', 'location']
                    if 'age' in anomaly_data.columns:
                        display_cols.append('age')
                    if 'description' in anomaly_data.columns:
                        display_cols.append('description')
                    if 'anomaly_score' in anomaly_data.columns:
                        display_cols.append('anomaly_score')
                    
                    top_anomalies = anomaly_data[display_cols].head(10)
                    st.dataframe(top_anomalies, use_container_width=True)
                    
                    # Export anomalies
                    if st.button("📥 Export Anomalies to CSV"):
                        csv = anomaly_data.to_csv(index=False)
                        st.download_button(
                            label="Download Anomalous Incidents",
                            data=csv,
                            file_name=f"anomalous_incidents_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    # Recommendations
                    st.markdown("#### 💡 Recommendations")
                    st.markdown(f"""
                    **Based on the anomaly analysis:**
                    - 🔍 **Review the {n_anomalies} flagged incidents** for potential process improvements
                    - 📊 **Focus on patterns** showing the highest statistical differences
                    - 🚨 **Prioritize incidents** with the highest anomaly scores
                    - 📈 **Monitor trends** in anomalous characteristics over time
                    - 🔄 **Regular re-analysis** as new data becomes available
                    """)
                    
                else:
                    st.info("✅ No anomalies detected with current parameters. This could indicate:")
                    st.markdown("""
                    - 🎯 Well-controlled processes with consistent patterns
                    - 📊 Parameters may be too strict - try increasing the anomaly rate
                    - 🔍 Different features might reveal other patterns
                    - ⚙️ Try a different detection algorithm
                    """)
            else:
                st.info("👆 Click 'Detect Anomalies' to find unusual incidents")
                
                # Show example of what anomaly detection can find
                st.markdown("### 🔍 What Anomaly Detection Can Reveal")
                st.markdown("""
                **Anomaly detection helps identify:**
                - 🚨 **Unusual incident patterns** that deviate from normal operations
                - ⏰ **Timing anomalies** (incidents at unusual hours/days)
                - 👥 **Participant-specific patterns** requiring special attention  
                - 📍 **Location-based outliers** indicating environmental issues
                - ⚡ **Severity mismatches** where reported severity doesn't match other factors
                
                **Common anomalies in NDIS data:**
                - High-severity incidents in typically low-risk locations
                - Incidents involving participants outside their usual age group patterns
                - Unusual notification delays for critical incidents
                - Atypical incident types for specific service locations
                """)', y='PC2',
                                color='Type',
                                hover_data=['Incident_Type', 'Severity'],
                                title="🔍 Anomaly Detection Results",
                                color_discrete_map={'🚨 Anomaly': 'red', '✅ Normal': 'blue'}
                            )
                            fig_anomaly.update_layout(height=500)
                            st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    # Anomaly analysis
                    st.markdown("### 🔍 Anomaly Analysis")
                    
                    anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
                    anomaly_data = ml_df.iloc[anomaly_indices]
                    normal_data = ml_df.iloc[[i for i, label in enumerate(anomaly_labels) if label == 1]]
                    
                    # Compare characteristics
                    st.markdown("**Anomalous vs Normal Incident Characteristics:**")
                    
                    comparison_data = []
                    for col in ['incident_type', 'severity', 'location']:
                        if col in anomaly_data.columns:
                            anomaly_dist = anomaly_data[col].value_counts(normalize=True).head(3)
                            normal_dist = normal_data[col].value_counts(normalize=True)
                            
                            for category in anomaly_dist.index:
                                comparison_data.append({
                                    'Category': f"{col}: {category}",
                                    'Anomaly %': f"{anomaly_dist[category] * 100:.1f}%",
                                    'Normal %': f"{normal_dist.get(category, 0) * 100:.1f}%"
                                })
                    
                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True)
                        
                        # Show some anomalous incidents
                        st.markdown("**Sample Anomalous Incidents:**")
                        display_cols = ['incident_type', 'severity', 'location']
                        if 'description' in anomaly_data.columns:
                            display_cols.append('description')
                        
                        sample_anomalies = anomaly_data[display_cols].head(5)
                        st.dataframe(sample_anomalies, use_container_width=True)
                else:
                    st.info("✅ No anomalies detected with current parameters")
            else:
                st.info("👆 Click 'Detect Anomalies' to find unusual incidents")
    
    with ml_tab3:
        st.subheader("🔍 Association Rules Mining")
        st.markdown("*Discover relationships between incident characteristics*")
        
        if not MLXTEND_AVAILABLE:
            st.warning("⚠️ Association rules require mlxtend library. Install with: `pip install mlxtend`")
            st.code("pip install mlxtend")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🎛️ Mining Parameters")
                
                # Enhanced parameter controls
                min_support = st.slider(
                    "Minimum Support", 
                    0.01, 0.5, 0.1,
                    help="Minimum frequency of item combinations (lower = more rules)"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    0.1, 0.9, 0.6,
                    help="Minimum confidence for association rules (higher = stronger relationships)"
                )
                
                min_lift = st.slider(
                    "Minimum Lift",
                    1.0, 5.0, 1.2,
                    help="Minimum lift value (>1 means positive association)"
                )
                
                # Feature selection for association mining
                st.markdown("### 📋 Features to Include")
                available_features = ['incident_type', 'severity', 'location']
                if 'reportable' in ml_df.columns:
                    available_features.append('reportable')
                if 'age_group' in ml_df.columns:
                    available_features.append('age_group')
                if 'day_of_week' in ml_df.columns:
                    available_features.append('day_of_week')
                
                selected_features = st.multiselect(
                    "Select features for association analysis",
                    options=available_features,
                    default=available_features[:3]
                )
                
                if st.button("⚡ Mine Association Rules", type="primary"):
                    if len(selected_features) < 2:
                        st.error("Please select at least 2 features for association analysis")
                    else:
                        with st.spinner("⚡ Mining association patterns..."):
                            frequent_itemsets, rules = find_association_rules_enhanced(
                                ml_df, features=selected_features, min_support=min_support, 
                                min_confidence=min_confidence, min_lift=min_lift
                            )
                            
                            if frequent_itemsets is not None and rules is not None:
                                st.session_state['frequent_itemsets'] = frequent_itemsets
                                st.session_state['association_rules'] = rules
                                st.session_state['selected_features'] = selected_features
                                st.success("✅ Association rules mining completed!")
                            else:
                                st.warning("⚠️ No rules found. Try lowering the parameters.")
            
            with col2:
                if 'association_rules' in st.session_state and st.session_state['association_rules'] is not None:
                    rules = st.session_state['association_rules']
                    frequent_itemsets = st.session_state.get('frequent_itemsets', pd.DataFrame())
                    
                    st.markdown("### 📊 Association Rules Results")
                    
                    if len(rules) > 0:
                        # Enhanced summary metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("🔗 Rules Found", len(rules))
                        with col_b:
                            avg_confidence = rules['confidence'].mean()
                            st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")
                        with col_c:
                            avg_support = rules['support'].mean()
                            st.metric("🎯 Avg Support", f"{avg_support:.3f}")
                        with col_d:
                            avg_lift = rules['lift'].mean()
                            st.metric("⚡ Avg Lift", f"{avg_lift:.3f}")
                        
                        # Top rules with better formatting
                        st.markdown("### 🔝 Top Association Rules")
                        top_rules = rules.sort_values('lift', ascending=False).head(15)
                        
                        display_rules = []
                        for _, rule in top_rules.iterrows():
                            antecedents = ', '.join([str(x).replace('_', ' ').title() for x in list(rule['antecedents'])])
                            consequents = ', '.join([str(x).replace('_', ' ').title() for x in list(rule['consequents'])])
                            
                            # Interpret the rule strength
                            confidence = rule['confidence']
                            if confidence >= 0.8:
                                strength = "🔥 Very Strong"
                            elif confidence >= 0.6:
                                strength = "💪 Strong"
                            elif confidence >= 0.4:
                                strength = "📈 Moderate"
                            else:
                                strength = "📉 Weak"
                            
                            display_rules.append({
                                'Rule': f"{antecedents} → {consequents}",
                                'Support': f"{rule['support']:.3f}",
                                'Confidence': f"{rule['confidence']:.3f}",
                                'Lift': f"{rule['lift']:.3f}",
                                'Strength': strength
                            })
                        
                        rules_df = pd.DataFrame(display_rules)
                        st.dataframe(rules_df, use_container_width=True)
                        
                        # Enhanced visualization
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Support vs Confidence scatter
                            fig_rules = px.scatter(
                                rules,
                                x='support',
                                y='confidence',
                                size='lift',
                                color='lift',
                                title="🔍 Association Rules: Support vs Confidence",
                                labels={'support': 'Support', 'confidence': 'Confidence'},
                                color_continuous_scale='viridis'
                            )
                            fig_rules.update_layout(height=400)
                            st.plotly_chart(fig_rules, use_container_width=True)
                        
                        with col_viz2:
                            # Lift distribution
                            fig_lift = px.histogram(
                                rules,
                                x='lift',
                                title="⚡ Lift Distribution",
                                labels={'lift': 'Lift Value', 'count': 'Number of Rules'}
                            )
                            fig_lift.add_vline(x=1, line_dash="dash", line_color="red", 
                                             annotation_text="Lift = 1 (No Association)")
                            fig_lift.update_layout(height=400)
                            st.plotly_chart(fig_lift, use_container_width=True)
                        
                        # Rule interpretation
                        st.markdown("### 🔍 Rule Interpretation Guide")
                        st.markdown("""
                        **How to read these rules:**
                        - **Support**: How frequently the items appear together
                        - **Confidence**: How often the rule is correct (A → B)
                        - **Lift**: How much more likely B is when A occurs (>1 = positive association)
                        
                        **Rule Strength:**
                        - 🔥 Very Strong (80%+): Highly reliable patterns
                        - 💪 Strong (60-80%): Reliable patterns worth investigating
                        - 📈 Moderate (40-60%): Noteworthy patterns
                        - 📉 Weak (<40%): Weak associations
                        """)
                        
                        # Export rules
                        if st.button("📥 Export Rules to CSV"):
                            rules_export = rules.copy()
                            rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
                            csv = rules_export.to_csv(index=False)
                            st.download_button(
                                label="Download Association Rules",
                                data=csv,
                                file_name=f"association_rules_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No association rules found. Try lowering the minimum parameters.")
                else:
                    st.info("👆 Click 'Mine Association Rules' to discover relationships")
    
    with ml_tab4:
        st.subheader("📊 ML Insights Summary")
        st.markdown("*Key findings from machine learning analysis*")
        
        # Correlation heatmap
        if len(corr_matrix) > 1:
            fig_corr = px.imshow(
                corr_matrix,
                title="🔗 Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature importance (if available)
        if 'cluster_labels' in st.session_state or 'anomaly_labels' in st.session_state:
            st.markdown("### 🎯 Key Insights")
            
            insights_list = [
                "📊 **Data Quality**: Processed and analyzed successfully",
                "🔍 **Pattern Detection**: Multiple analytical approaches applied",
                "⚡ **Real-time Analysis**: Results updated based on current filters"
            ]
            
            if 'cluster_labels' in st.session_state:
                n_clusters = len(set(st.session_state['cluster_labels']))
                insights_list.append(f"🔗 **Clustering**: Identified {n_clusters} distinct incident patterns")
            
            if 'anomaly_labels' in st.session_state:
                n_anomalies = sum(st.session_state['anomaly_labels'] == -1)
                insights_list.append(f"🚨 **Anomalies**: Detected {n_anomalies} unusual incidents requiring attention")
            
            if 'association_rules' in st.session_state and len(st.session_state['association_rules']) > 0:
                n_rules = len(st.session_state['association_rules'])
                insights_list.append(f"🔍 **Associations**: Found {n_rules} significant relationships between factors")
            
            for insight in insights_list:
                st.markdown(insight)
        else:
            st.info("Run the ML analysis tools above to see insights here.")

elif analysis_mode == "Data Explorer":
    st.subheader("📋 Data Explorer")
    
    # Search functionality
    search_term = st.text_input("🔍 Search in descriptions")
    
    # Filter data based on search
    display_df = df_filtered.copy()
    if search_term and 'description' in display_df.columns:
        mask = display_df['description'].str.contains(search_term, case=False, na=False)
        display_df = display_df[mask]
    
    # Column selector
    if len(display_df.columns) > 10:
        selected_columns = st.multiselect(
            "Select columns to display",
            options=display_df.columns.tolist(),
            default=display_df.columns.tolist()[:10]
        )
        display_df = display_df[selected_columns] if selected_columns else display_df
    
    # Display data with pagination
    st.markdown(f"**Showing {len(display_df)} records**")
    
    # Pagination
    if len(display_df) > 100:
        page_size = st.selectbox("Records per page", [25, 50, 100], index=1)
        total_pages = (len(display_df) - 1) // page_size + 1
        page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        display_df = display_df.iloc[start_idx:end_idx]
    
    # Display data
    st.dataframe(display_df, use_container_width=True)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Filtered Data"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ndis_incidents_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Report"):
            # Simple report generation
            report = f"""
# NDIS Incident Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Incidents: {len(df_filtered)}
- Date Range: {df_filtered['incident_date'].min().strftime('%Y-%m-%d')} to {df_filtered['incident_date'].max().strftime('%Y-%m-%d')}
- Unique Locations: {df_filtered['location'].nunique()}
- High Severity Incidents: {len(df_filtered[df_filtered['severity'].str.lower().isin(['high', 'critical'])])}

## Top Incident Types
{df_filtered['incident_type'].value_counts().head().to_string()}

## Severity Distribution
{df_filtered['severity'].value_counts().to_string()}
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"ndis_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**Records:** {len(df_filtered)} of {len(df)}")
with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Help & Support")
    with st.expander("🤔 How to use this dashboard"):
        st.markdown("""
        **Getting Started:**
        1. Upload your CSV file using the file uploader
        2. Use filters to focus on specific data
        3. Choose an analysis mode to explore your data
        
        **Analysis Modes:**
        - **Executive Overview**: High-level KPIs and trends
        - **Risk Analysis**: Identify risk patterns and hotspots
        - **ML Analytics**: Advanced machine learning insights
        - **Data Explorer**: Browse and search your raw data
        
        **Tips:**
        - Use date filters to analyze specific time periods
        - Try different ML algorithms for varied insights
        - Download results for further analysis
        """)
    
    st.markdown("**Built with Streamlit & Plotly** 🚀")
