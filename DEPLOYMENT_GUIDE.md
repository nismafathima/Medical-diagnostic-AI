# 🚀 MedAI Deployment Guide - CPU Optimized

## Quick Navigation
- [Local Deployment](#-local-deployment)
- [Cloud Deployment](#-cloud-deployment)
- [Docker Deployment](#-docker-deployment)
- [Production Deployment](#-production-deployment)

---

## 💻 Local Deployment

### Windows 10/11

**Prerequisites:**
- Python 3.10+ installed
- 8GB+ RAM available
- 2GB free storage

**Steps:**

1. **Download & Extract**
   ```bash
   # Download medai_optimized.zip
   # Extract to C:\Users\YourName\medai
   cd C:\Users\YourName\medai
   ```

2. **Run Setup**
   ```bash
   # Double-click setup.bat
   # OR in Command Prompt:
   setup.bat
   ```

3. **Run Application**
   ```bash
   # Activate virtual environment (if not done by setup)
   venv\Scripts\activate
   
   # Start application
   streamlit run app.py
   ```

4. **Open Browser**
   ```
   http://localhost:8501
   ```

**Troubleshooting:**
- If setup fails: Run `python -m pip install -r requirements.txt` manually
- If streamlit not found: Check Python path with `python --version`
- If models won't download: Check internet connection

---

### macOS 10.14+

**Prerequisites:**
- Python 3.10+ (use Homebrew: `brew install python@3.10`)
- 8GB+ RAM
- 2GB free storage
- M1/M2: Native support, optimized

**Steps:**

1. **Download & Extract**
   ```bash
   cd ~/Downloads
   unzip medai_optimized.zip
   cd medai
   ```

2. **Run Setup**
   ```bash
   chmod +x setup.sh
   bash setup.sh
   ```

3. **Activate Environment**
   ```bash
   source venv/bin/activate
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

5. **Open Browser**
   ```
   http://localhost:8501
   ```

**M1/M2 Optimization:**
```bash
# M1/M2 chips are highly optimized for PyTorch
# Expect 20-30% faster performance than Intel x86
# No special configuration needed
```

---

### Linux (Ubuntu 20.04+)

**Prerequisites:**
- Python 3.10+ (`sudo apt install python3.10`)
- 8GB+ RAM
- 2GB free storage

**Steps:**

1. **Download & Extract**
   ```bash
   cd ~
   unzip medai_optimized.zip
   cd medai
   ```

2. **Run Setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate Environment**
   ```bash
   source venv/bin/activate
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

5. **Access Application**
   ```
   http://localhost:8501
   ```

**System Dependencies (if needed):**
```bash
# Audio support (for speech recognition)
sudo apt install libsndfile1

# Image processing
sudo apt install libmagick++-dev

# Optional: hardware acceleration
sudo apt install libopenblas-dev
```

---

## ☁️ Cloud Deployment

### Google Colab (Free, Easiest)

**Advantages:**
- ✅ Free computing
- ✅ No installation needed
- ✅ Pre-installed packages
- ✅ 12+ hours sessions
- ✅ Good for testing

**Steps:**

1. **Open Google Colab**
   ```
   https://colab.research.google.com
   ```

2. **Clone Repository**
   ```python
   !git clone https://github.com/medai/medai.git
   %cd medai
   ```

3. **Install Dependencies**
   ```python
   !pip install -r requirements.txt
   ```

4. **Run Streamlit with Pyngrok**
   ```python
   # Install ngrok for tunneling
   !pip install pyngrok
   
   # Set ngrok token (from https://dashboard.ngrok.com)
   from pyngrok import ngrok
   
   # Run Streamlit
   !streamlit run app.py &
   
   # Create tunnel
   public_url = ngrok.connect(8501)
   print(public_url)
   ```

5. **Access via Public URL**
   ```
   https://your-ngrok-url.ngrok.io
   ```

**Limitations:**
- ⚠️ Free tier: CPU only (but optimized!)
- ⚠️ Session timeout after 12 hours
- ⚠️ Slower than local (shared resources)

---

### Streamlit Cloud (Recommended)

**Advantages:**
- ✅ Purpose-built for Streamlit
- ✅ Free tier available
- ✅ Custom domain support
- ✅ Automatic deploys
- ✅ GitHub integration

**Steps:**

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "CPU optimized medai"
   git push origin main
   ```

2. **Create Streamlit Account**
   ```
   https://share.streamlit.io
   ```

3. **Deploy**
   - Click "New app"
   - Connect GitHub repo
   - Select branch: `main`
   - Select script: `app.py`
   - Click "Deploy"

4. **Configure (streamlit/config.toml)**
   ```toml
   [server]
   headless = true
   port = 8501
   
   [logger]
   level = "error"
   ```

5. **Access App**
   ```
   https://your-username-medai.streamlit.app
   ```

**Costs:**
- Free: 1 concurrent user
- Pro: $5/month, 5 concurrent users
- Business: Custom pricing

---

### AWS EC2 (Advanced)

**Advantages:**
- ✅ Full control
- ✅ Scalable
- ✅ Custom domain
- ✅ High availability

**Steps:**

1. **Launch EC2 Instance**
   ```
   AMI: Ubuntu 20.04 LTS (free tier eligible)
   Type: t3.medium (2 cores, 4GB RAM)
   Storage: 20GB
   ```

2. **Connect via SSH**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Setup System**
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv
   ```

4. **Deploy App**
   ```bash
   git clone https://github.com/medai/medai.git
   cd medai
   ./setup.sh
   source venv/bin/activate
   streamlit run app.py
   ```

5. **Setup Reverse Proxy (Nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
       }
   }
   ```

6. **Enable HTTPS (Let's Encrypt)**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

**Costs:**
- Free tier: 12 months
- After: ~$10-50/month depending on usage

---

### Heroku (Simple & Quick)

**Advantages:**
- ✅ 5-minute deployment
- ✅ Free tier available
- ✅ Automatic scaling
- ✅ Custom domain

**Setup:**

1. **Create Procfile**
   ```
   web: sh setup.sh && streamlit run app.py --logger.level=error
   ```

2. **Create requirements.txt**
   ```
   # Already done in repo
   ```

3. **Deploy**
   ```bash
   heroku login
   heroku create medai-yourname
   git push heroku main
   ```

4. **Access**
   ```
   https://medai-yourname.herokuapp.com
   ```

**Costs:**
- Free: Limited (dyno hours)
- Eco: $5/month, always on
- Professional: $7-50/month

---

### DigitalOcean App Platform

**Advantages:**
- ✅ $5/month starting
- ✅ Simple deployment
- ✅ Good uptime
- ✅ GitHub integration

**Steps:**

1. **Create App**
   - Go to DigitalOcean Console
   - Apps → Create App
   - Connect GitHub repo

2. **Configure**
   ```yaml
   name: medai
   services:
   - name: web
     github:
       repo: your/medai
       branch: main
     build_command: pip install -r requirements.txt
     run_command: streamlit run app.py
   ```

3. **Deploy**
   - Set environment variables
   - Click "Deploy"

4. **Access**
   ```
   https://medai-<random>.ondigitalocean.app
   ```

---

## 🐳 Docker Deployment

### Docker Setup

**Create Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create cache directory
RUN mkdir -p ~/.cache/huggingface

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD streamlit run app.py --logger.level=error
```

**Create docker-compose.yml:**

```yaml
version: '3.8'

services:
  medai:
    build: .
    container_name: medai_app
    ports:
      - "8501:8501"
    volumes:
      - ./cache:/root/.cache/huggingface
      - ./data:/app/data
    environment:
      - TORCH_NUM_THREADS=4
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Build & Run:**

```bash
# Build image
docker build -t medai:latest .

# Run container
docker run -p 8501:8501 medai:latest

# Or with docker-compose
docker-compose up -d
```

**Stop Container:**

```bash
docker-compose down
# or
docker stop medai_app
```

---

### Docker Hub Deployment

**Push to Docker Hub:**

```bash
# Login
docker login

# Tag image
docker tag medai:latest yourusername/medai:latest

# Push
docker push yourusername/medai:latest

# Pull and run
docker run -p 8501:8501 yourusername/medai:latest
```

---

## 🏢 Production Deployment

### Recommended Setup

**Architecture:**
```
Internet → Load Balancer → Nginx (Reverse Proxy)
         → Docker Container 1
         → Docker Container 2
         → Docker Container 3
         ↓
       Cache (Redis) → Model Cache
       ↓
    Database (PostgreSQL) → Logs & Analytics
```

### Production Checklist

- [ ] Environment variables configured
- [ ] HTTPS/SSL enabled
- [ ] Load balancer configured
- [ ] Monitoring & alerting setup
- [ ] Logging configured
- [ ] Backup strategy
- [ ] Disaster recovery plan
- [ ] Performance monitoring
- [ ] Security audit completed
- [ ] Rate limiting configured

### Performance Optimization

**For Production:**

1. **Nginx Configuration**
   ```nginx
   upstream medai {
       server app1:8501;
       server app2:8501;
       server app3:8501;
   }
   
   server {
       listen 443 ssl http2;
       server_name medai.example.com;
       
       # SSL configuration
       ssl_certificate /etc/ssl/certs/cert.pem;
       ssl_certificate_key /etc/ssl/private/key.pem;
       
       # Caching
       proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=medai:10m;
       
       location / {
           proxy_pass http://medai;
           proxy_cache medai;
           proxy_cache_valid 200 1h;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **Resource Limits**
   ```yaml
   resources:
     limits:
       cpus: '2'
       memory: 4G
     reservations:
       cpus: '1'
       memory: 2G
   ```

3. **Auto-scaling**
   ```yaml
   autoscaling:
     min_replicas: 2
     max_replicas: 5
     target_cpu_utilization: 70%
   ```

---

## 🔐 Security Checklist

### Before Production

- [ ] Environment variables set (API keys)
- [ ] HTTPS/SSL configured
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Input validation enabled
- [ ] SQL injection protection
- [ ] XSS protection enabled
- [ ] CSRF tokens enabled
- [ ] Security headers configured

### Monitoring & Logging

```bash
# Container logs
docker logs -f medai_app

# System metrics
docker stats medai_app

# Application logs
tail -f logs/app.log

# Performance monitoring
# Use Prometheus + Grafana
```

---

## 📊 Deployment Comparison

| Platform | Cost | Setup | Uptime | Scalability |
|----------|------|-------|--------|-------------|
| **Local** | Free | 10min | N/A | N/A |
| **Colab** | Free | 5min | 12h | ❌ |
| **Streamlit Cloud** | Free-5$/mo | 5min | 99.9% | ✅ Good |
| **Heroku** | Free-50$/mo | 5min | 99.9% | ✅ Good |
| **AWS EC2** | 10-50$/mo | 30min | 99.99% | ✅ Excellent |
| **DigitalOcean** | 5-50$/mo | 10min | 99.99% | ✅ Good |
| **Docker** | Variable | 20min | 99.9%+ | ✅ Excellent |

---

## 🆘 Troubleshooting Deployments

### Issue: Slow Loading
- **Cause:** Model downloads
- **Solution:** Pre-download models, increase timeout

### Issue: Memory Error
- **Cause:** Insufficient RAM
- **Solution:** Increase container memory limits

### Issue: Models Not Found
- **Cause:** Cache not persistent
- **Solution:** Use volumes for model cache

### Issue: High CPU Usage
- **Cause:** Normal during inference
- **Solution:** Monitor, adjust scaling

### Issue: Port Already in Use
- **Cause:** Another service on port 8501
- **Solution:** `docker ps`, then `docker stop <container>`

---

## 📱 Mobile Access

### Remote Access on Same Network
```bash
# Get your machine IP
ipconfig getifaddr en0  # macOS
hostname -I  # Linux
ipconfig  # Windows

# Access from phone on same network
http://your-ip:8501
```

### Remote Access (Internet)
- Use ngrok (temporary)
- Use cloud deployment (permanent)
- Use VPN (secure)

---

## 📞 Support

- **Installation:** See README_OPTIMIZED.md
- **Performance:** See OPTIMIZATION_GUIDE.md
- **Bugs:** Check BUG_FIXES_LOG.md
- **Issues:** GitHub Issues
- **Discussion:** Community Forum

---

**Last Updated:** July 2024  
**Version:** 2.0  
**Status:** ✅ Production Ready
