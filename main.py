import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import re
from typing import List, Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go

class RecordLinker:
    def __init__(self, threshold_similarity=0.8, date_tolerance_days=5, amount_tolerance_pct=0.5):
        self.threshold_similarity = threshold_similarity
        self.date_tolerance_days = date_tolerance_days
        self.amount_tolerance_pct = amount_tolerance_pct
    
    def normalize_identifier(self, identifier):
        """Extract meaningful parts from various identifier formats"""
        if pd.isna(identifier):
            return ""
        
        identifier = str(identifier).upper().strip()
        
        # Handle scientific notation (like 2.03E+09)
        if 'E+' in identifier:
            try:
                # Convert scientific notation to integer string
                num = int(float(identifier))
                return str(num)
            except:
                pass
        
        # Extract digits for comparison
        digits_only = re.sub(r'\D', '', identifier)
        
        # Extract meaningful tokens
        tokens = re.findall(r'[A-Z]+|\d+', identifier)
        
        return {
            'original': identifier,
            'digits_only': digits_only,
            'tokens': tokens,
            'normalized': re.sub(r'[^A-Z0-9]', '', identifier)
        }
    
    def identifier_similarity(self, id1, id2):
        """Calculate similarity between two identifiers"""
        norm1 = self.normalize_identifier(id1)
        norm2 = self.normalize_identifier(id2)
        
        if isinstance(norm1, str) or isinstance(norm2, str):
            return 0.0
        
        # Exact match
        if norm1['original'] == norm2['original']:
            return 1.0
        
        # Digits-only match
        if norm1['digits_only'] and norm2['digits_only']:
            if norm1['digits_only'] == norm2['digits_only']:
                return 0.9
            # Partial digits match
            if len(norm1['digits_only']) >= 6 and len(norm2['digits_only']) >= 6:
                similarity = SequenceMatcher(None, norm1['digits_only'], norm2['digits_only']).ratio()
                if similarity > 0.8:
                    return 0.7 * similarity
        
        # Token-based similarity
        if norm1['tokens'] and norm2['tokens']:
            common_tokens = set(norm1['tokens']) & set(norm2['tokens'])
            all_tokens = set(norm1['tokens']) | set(norm2['tokens'])
            if all_tokens:
                return 0.6 * len(common_tokens) / len(all_tokens)
        
        # Normalized string similarity
        return 0.4 * SequenceMatcher(None, norm1['normalized'], norm2['normalized']).ratio()
    
    def name_similarity(self, name1, name2):
        """Calculate similarity between customer names"""
        if pd.isna(name1) or pd.isna(name2):
            return 0.0
        
        name1 = str(name1).lower().strip()
        name2 = str(name2).lower().strip()
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Split names and find common parts
        parts1 = set(name1.split())
        parts2 = set(name2.split())
        
        if parts1 and parts2:
            common = parts1 & parts2
            total = parts1 | parts2
            return len(common) / len(total)
        
        return SequenceMatcher(None, name1, name2).ratio()
    
    def email_similarity(self, email1, email2):
        """Calculate similarity between emails"""
        if pd.isna(email1) or pd.isna(email2):
            return 0.0
        
        email1 = str(email1).lower().strip()
        email2 = str(email2).lower().strip()
        
        if email1 == email2:
            return 1.0
        
        return SequenceMatcher(None, email1, email2).ratio()
    
    def amount_similarity(self, amount1, amount2):
        """Calculate similarity between amounts"""
        try:
            amt1 = float(amount1) if not pd.isna(amount1) else 0
            amt2 = float(amount2) if not pd.isna(amount2) else 0
            
            if amt1 == amt2:
                return 1.0
            
            if amt1 == 0 or amt2 == 0:
                return 0.0
            
            # Calculate percentage difference
            diff_pct = abs(amt1 - amt2) / max(amt1, amt2) * 100
            
            if diff_pct <= self.amount_tolerance_pct:
                return 1.0 - (diff_pct / self.amount_tolerance_pct) * 0.2
            
            return max(0, 1.0 - (diff_pct / 10))  # Gradual decrease
        except:
            return 0.0
    
    def date_similarity(self, date1, date2):
        """Calculate similarity between dates"""
        try:
            # Parse dates with various formats
            for date_format in ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    d1 = datetime.strptime(str(date1), date_format)
                    break
                except:
                    continue
            else:
                return 0.0
                
            for date_format in ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    d2 = datetime.strptime(str(date2), date_format)
                    break
                except:
                    continue
            else:
                return 0.0
            
            diff_days = abs((d1 - d2).days)
            
            if diff_days == 0:
                return 1.0
            elif diff_days <= self.date_tolerance_days:
                return 1.0 - (diff_days / self.date_tolerance_days) * 0.3
            else:
                return max(0, 1.0 - (diff_days / 30))  # Gradual decrease over a month
        except:
            return 0.0
    
    def calculate_match_score(self, record1, record2, field_mapping):
        """Calculate overall match score between two records"""
        scores = {}
        weights = {
            'identifier': 0.3,
            'amount': 0.25,
            'date': 0.2,
            'name': 0.15,
            'email': 0.1
        }
        
        # Identifier similarity
        id_score = self.identifier_similarity(
            record1[field_mapping['source_a']['id']], 
            record2[field_mapping['source_b']['id']]
        )
        scores['identifier'] = id_score
        
        # Amount similarity (using total_amount)
        amt_score = self.amount_similarity(
            record1[field_mapping['source_a']['total_amount']], 
            record2[field_mapping['source_b']['total_amount']]
        )
        scores['amount'] = amt_score
        
        # Date similarity
        date_score = self.date_similarity(
            record1[field_mapping['source_a']['date']], 
            record2[field_mapping['source_b']['date']]
        )
        scores['date'] = date_score
        
        # Name similarity
        name_score = self.name_similarity(
            record1[field_mapping['source_a']['name']], 
            record2[field_mapping['source_b']['name']]
        )
        scores['name'] = name_score
        
        # Email similarity
        email_score = self.email_similarity(
            record1[field_mapping['source_a']['email']], 
            record2[field_mapping['source_b']['email']]
        )
        scores['email'] = email_score
        
        # Calculate weighted total
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return total_score, scores
    
    def find_matches(self, df_a, df_b, field_mapping):
        """Find matches between two dataframes"""
        matches = []
        
        for idx_a, record_a in df_a.iterrows():
            best_score = 0
            best_match = None
            best_scores_detail = None
            
            for idx_b, record_b in df_b.iterrows():
                score, scores_detail = self.calculate_match_score(record_a, record_b, field_mapping)
                
                if score > best_score and score >= self.threshold_similarity:
                    best_score = score
                    best_match = idx_b
                    best_scores_detail = scores_detail
            
            matches.append({
                'source_a_idx': idx_a,
                'source_b_idx': best_match,
                'match_score': best_score,
                'scores_detail': best_scores_detail,
                'is_match': best_match is not None
            })
        
        return matches

def main():
    st.set_page_config(page_title="Cross-Source Record Linking", layout="wide")
    
    st.title("🔗 Cross-Source Record Linking")
    st.write("Upload two CSV files to find matching records across different data sources.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.8, 
        step=0.05,
        help="Minimum similarity score to consider a match"
    )
    
    date_tolerance = st.sidebar.slider(
        "Date Tolerance (days)", 
        min_value=0, 
        max_value=30, 
        value=5,
        help="Maximum days difference for date matching"
    )
    
    amount_tolerance = st.sidebar.slider(
        "Amount Tolerance (%)", 
        min_value=0.0, 
        max_value=5.0, 
        value=0.5, 
        step=0.1,
        help="Maximum percentage difference for amount matching"
    )
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source A (Ground Truth)")
        file_a = st.file_uploader("Upload Source A CSV", type=['csv'], key="file_a")
    
    with col2:
        st.subheader("Source B (Noisy Data)")
        file_b = st.file_uploader("Upload Source B CSV", type=['csv'], key="file_b")
    
    if file_a and file_b:
        # Read files
        try:
            df_a = pd.read_csv(file_a)
            df_b = pd.read_csv(file_b)
            
            st.success(f"Loaded Source A: {len(df_a)} records, Source B: {len(df_b)} records")
            
            # Field mapping (hardcoded based on the provided data structure)
            field_mapping = {
                'source_a': {
                    'id': 'invoice_id',
                    'name': 'customer_name',
                    'email': 'customer_email',
                    'total_amount': 'total_amount',
                    'date': 'invoice_date'
                },
                'source_b': {
                    'id': 'ref_code',
                    'name': 'client',
                    'email': 'email',
                    'total_amount': 'grand_total',
                    'date': 'doc_date'
                }
            }
            
            # Show data preview
            with st.expander("Data Preview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Source A Sample:**")
                    st.dataframe(df_a.head())
                with col2:
                    st.write("**Source B Sample:**")
                    st.dataframe(df_b.head())
            
            # Perform matching
            if st.button("🔍 Find Matches", type="primary"):
                with st.spinner("Finding matches..."):
                    linker = RecordLinker(
                        threshold_similarity=similarity_threshold,
                        date_tolerance_days=date_tolerance,
                        amount_tolerance_pct=amount_tolerance
                    )
                    
                    matches = linker.find_matches(df_a, df_b, field_mapping)
                    
                    # Process results
                    matched_count = sum(1 for m in matches if m['is_match'])
                    unmatched_count = len(matches) - matched_count
                    
                    # Display summary
                    st.subheader("📊 Matching Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records (Source A)", len(df_a))
                    with col2:
                        st.metric("Matched Records", matched_count)
                    with col3:
                        st.metric("Unmatched Records", unmatched_count)
                    with col4:
                        st.metric("Match Rate", f"{matched_count/len(df_a)*100:.1f}%")
                    
                    # Create results dataframe
                    results = []
                    for match in matches:
                        if match['is_match']:
                            row_a = df_a.iloc[match['source_a_idx']]
                            row_b = df_b.iloc[match['source_b_idx']]
                            
                            results.append({
                                'Source A ID': row_a[field_mapping['source_a']['id']],
                                'Source B ID': row_b[field_mapping['source_b']['id']],
                                'Customer Name A': row_a[field_mapping['source_a']['name']],
                                'Customer Name B': row_b[field_mapping['source_b']['name']],
                                'Amount A': row_a[field_mapping['source_a']['total_amount']],
                                'Amount B': row_b[field_mapping['source_b']['total_amount']],
                                'Date A': row_a[field_mapping['source_a']['date']],
                                'Date B': row_b[field_mapping['source_b']['date']],
                                'Match Score': f"{match['match_score']:.3f}",
                                'ID Score': f"{match['scores_detail']['identifier']:.3f}",
                                'Amount Score': f"{match['scores_detail']['amount']:.3f}",
                                'Date Score': f"{match['scores_detail']['date']:.3f}",
                                'Name Score': f"{match['scores_detail']['name']:.3f}",
                                'Email Score': f"{match['scores_detail']['email']:.3f}"
                            })
                        else:
                            row_a = df_a.iloc[match['source_a_idx']]
                            results.append({
                                'Source A ID': row_a[field_mapping['source_a']['id']],
                                'Source B ID': 'NO MATCH',
                                'Customer Name A': row_a[field_mapping['source_a']['name']],
                                'Customer Name B': 'NO MATCH',
                                'Amount A': row_a[field_mapping['source_a']['total_amount']],
                                'Amount B': 'NO MATCH',
                                'Date A': row_a[field_mapping['source_a']['date']],
                                'Date B': 'NO MATCH',
                                'Match Score': '0.000',
                                'ID Score': '0.000',
                                'Amount Score': '0.000',
                                'Date Score': '0.000',
                                'Name Score': '0.000',
                                'Email Score': '0.000'
                            })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results table
                    st.subheader("🔗 Detailed Matching Results")
                    
                    # Add filters
                    col1, col2 = st.columns(2)
                    with col1:
                        show_matches_only = st.checkbox("Show matches only", value=False)
                    with col2:
                        min_score_filter = st.slider("Minimum score to display", 0.0, 1.0, 0.0, 0.1)
                    
                    # Filter results
                    filtered_df = results_df.copy()
                    if show_matches_only:
                        filtered_df = filtered_df[filtered_df['Source B ID'] != 'NO MATCH']
                    
                    filtered_df = filtered_df[filtered_df['Match Score'].astype(float) >= min_score_filter]
                    
                    # Color code the dataframe
                    def color_match_score(val):
                        if val == 'NO MATCH' or val == '0.000':
                            return 'background-color: #ffebee'
                        score = float(val)
                        if score >= 0.9:
                            return 'background-color: #e8f5e8'
                        elif score >= 0.7:
                            return 'background-color: #fff3e0'
                        else:
                            return 'background-color: #fce4ec'
                    
                    styled_df = filtered_df.style.applymap(
                        color_match_score, 
                        subset=['Match Score']
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Visualizations
                    st.subheader("📈 Match Score Distribution")
                    
                    # Score distribution
                    scores = [float(m['match_score']) for m in matches if m['match_score'] > 0]
                    if scores:
                        fig = px.histogram(
                            x=scores, 
                            nbins=20, 
                            title="Distribution of Match Scores",
                            labels={'x': 'Match Score', 'y': 'Frequency'}
                        )
                        fig.add_vline(x=similarity_threshold, line_dash="dash", 
                                     annotation_text=f"Threshold ({similarity_threshold})")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Export results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"record_linking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()