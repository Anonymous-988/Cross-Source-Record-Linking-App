# 🔗 FinKraft Cross-Source Record Linking Challenge

## 📋 Project Overview

This Streamlit application solves **Project 7 - Cross-Source Record Linking** for FinKraft's data challenge. The system intelligently matches overlapping invoice records between two data sources despite varying identifiers, date drift, minor rounding differences, and injected ambiguities.

### Challenge Description
- **Objective**: Link records between Project7SourceA.csv (ground-truth-ish) and Project7SourceB.csv (noisy)
- **Complexity**: Handle exact, digits-only, embedded tokens, prefixes in identifiers
- **Variations**: Date drift, minor amount rounding, customer name variations
- **Tie-breakers**: Purchase orders, customer details, email matching

## 🎯 Key Features

### Smart Identifier Matching
- **Exact Matching**: Direct identifier comparison (highest confidence)
- **Core Number Extraction**: Handles embedded tokens in formats like `INV-2025XXXXXX` vs `REF-XXX-XXX`
- **Scientific Notation**: Converts formats like `2.03E+09` to standard numbers
- **Pattern Recognition**: Identifies and matches `REF-XXX-XXX` patterns
- **Year-Context Matching**: Considers year prefixes for better accuracy

### Robust Financial Data Handling
- **Multi-Amount Validation**: Compares both net and total amounts
- **Configurable Tolerance**: Handles minor rounding differences (default 1%)
- **Currency Consistency**: Maintains currency context in matching

### Advanced Scoring System
```
Weighted Scoring:
├── Identifier Match: 35% (Primary identifier)
├── Amount Match: 25% (Financial validation)
├── Purchase Order: 15% (Strong tie-breaker)
├── Date Match: 15% (Accounting for processing delays)
├── Customer Name: 7% (Validation field)
└── Email: 3% (Secondary validation)
```

### Intelligent Deduplication
- **One-to-One Matching**: Prevents duplicate matches
- **Best Score Selection**: Prioritizes highest confidence matches
- **Global Optimization**: Ensures optimal matching across entire dataset

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit pandas numpy plotly difflib
```

### Installation & Setup
1. **Clone/Download** the application file
2. **Prepare Data**: Ensure you have `Project7SourceA.csv` and `Project7SourceB.csv`
3. **Run Application**:
   ```bash
   streamlit run record_linking_app.py
   ```

### Quick Start
1. **Upload Files**: Use the file uploaders for both source files
2. **Configure Parameters**: Adjust similarity threshold, date tolerance, amount tolerance
3. **Process**: Click "🔍 Find Matches" to start matching
4. **Review Results**: Analyze matches with detailed scoring
5. **Export**: Download results as CSV for further analysis

## 📊 Data Schema

### Source A (Ground Truth) - Project7SourceA.csv
```
invoice_id, po_number, customer_name, customer_email, amount, 
tax_amount, total_amount, currency, invoice_date
```

### Source B (Noisy Data) - Project7SourceB.csv
```
ref_code, purchase_order, client, email, net, tax, 
grand_total, ccy, doc_date
```

## 🔧 Configuration Options

### Similarity Threshold (Default: 0.75)
- **Range**: 0.5 - 1.0
- **Purpose**: Minimum confidence score to consider a match
- **Recommendation**: 0.75 for balanced precision/recall

### Date Tolerance (Default: 7 days)
- **Range**: 0 - 30 days
- **Purpose**: Account for processing delays and date drift
- **Use Case**: Invoice processing vs. system entry delays

### Amount Tolerance (Default: 1.0%)
- **Range**: 0 - 5%
- **Purpose**: Handle minor rounding differences
- **Typical**: Tax calculations, currency conversions

## 📈 Performance Metrics

### Matching Algorithm Confidence Levels
- **🟢 0.90-1.00**: High Confidence (Core number/exact match)
- **🟡 0.75-0.89**: Medium Confidence (Strong similarity)
- **🟠 0.60-0.74**: Low Confidence (Partial match)
- **🔴 <0.60**: No Match (Below threshold)

### Expected Results
Based on typical invoice matching scenarios:
- **Match Rate**: 85-95%
- **False Positives**: <2%
- **Processing Time**: <30 seconds for 1000 records

## 🎨 User Interface Features

### Interactive Dashboard
- **Real-time Preview**: View data samples before processing
- **Color-coded Results**: Visual confidence indicators
- **Detailed Scoring**: Component-wise match scores
- **Interactive Filtering**: Filter by match status and scores

### Visualization
- **Score Distribution**: Histogram of match confidence
- **Threshold Visualization**: Visual threshold line
- **Export Capabilities**: CSV download with timestamps

## 🧠 Algorithm Deep Dive

### Identifier Matching Strategy
1. **Preprocessing**: Normalize identifiers, extract patterns
2. **Core Number Extraction**: Remove prefixes, isolate meaningful digits
3. **Pattern Recognition**: Identify format types (INV-, REF-, scientific)
4. **Similarity Calculation**: Multi-level comparison with weighted scores

### Scoring Components
```python
# Example scoring breakdown
{
    'identifier': 0.92,      # Strong ID match
    'amount': 0.98,          # Exact amount match
    'purchase_order': 1.00,  # Perfect PO match
    'date': 0.85,            # 2-day difference
    'name': 0.95,            # Minor name variation
    'email': 1.00            # Exact email match
}
# Final Score: 0.934 (weighted average)
```

## 🚨 Challenge-Specific Optimizations

### FinKraft Invoice Patterns
- **INV-2025XXXXXX**: Standard invoice format
- **REF-XXX-XXX**: Reference code format
- **2.03E+09**: Numbers in E Notation.
- **PO-XXXX**: Purchase order matching as tie-breaker

### Date Drift Handling
- **Processing Delays**: 7-day default tolerance
- **Format Variations**: Multiple date format support
- **Business Logic**: Considers typical invoice-to-payment cycles

### Amount