/**
 * Sezgen — Havalimanı Pist FOD Tespit Sistemi
 * İstemci Tarafı Kontrol, Canlı Akış, Sistem Günlükleri ve Eğitim Grafikleri Motoru
 */

(() => {
    const htmlEl = document.documentElement;
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeToggleLabel = document.getElementById("themeToggleLabel");

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadText = document.getElementById("uploadText");

    const modeSwitchBtn = document.getElementById("modeSwitchBtn");
    const sliderEsik1 = document.getElementById("sliderEsik1");
    const sliderEsik2 = document.getElementById("sliderEsik2");
    const sliderEsik3 = document.getElementById("sliderEsik3");
    const valEsik1 = document.getElementById("valEsik1");
    const valEsik2 = document.getElementById("valEsik2");
    const valEsik3 = document.getElementById("valEsik3");

    const startBtn = document.getElementById("startBtn");
    const hudTimestamp = document.getElementById("hudTimestamp");
    const hudBadgeText = document.getElementById("hudBadgeText");
    const alarmBanner = document.getElementById("alarmBanner");
    const logsContainer = document.getElementById("logsContainer");
    const foundItemsList = document.getElementById("foundItemsList");

    const chartLeftCanvas = document.getElementById("trainingChartLeft");
    const chartRightCanvas = document.getElementById("trainingChartRight");

    let isRunning = false;
    let isDeepScan = false;
    let cachedMetrics = null;
    let settingsDebounce = null;

    // 1. Tema Yönetimi (Aydınlık tema / Karanlık tema)
    function applyTheme(theme) {
        htmlEl.setAttribute("data-theme", theme);
        localStorage.setItem("sezgen_theme", theme);
        themeToggleLabel.textContent = theme === "dark" ? "Karanlık tema" : "Aydınlık tema";
        if (cachedMetrics) {
            renderBothCharts(cachedMetrics);
        }
    }

    const savedTheme = localStorage.getItem("sezgen_theme") || "light";
    applyTheme(savedTheme);

    themeToggleBtn.addEventListener("click", () => {
        const current = htmlEl.getAttribute("data-theme") === "dark" ? "dark" : "light";
        applyTheme(current === "dark" ? "light" : "dark");
    });

    // 2. Canlı UTC Saat Gösterimi (HUD)
    function updateHudClock() {
        const now = new Date();
        const yy = String(now.getUTCFullYear()).slice(-2);
        const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
        const dd = String(now.getUTCDate()).padStart(2, "0");
        const hh = String(now.getUTCHours()).padStart(2, "0");
        const mi = String(now.getUTCMinutes()).padStart(2, "0");
        const ss = String(now.getUTCSeconds()).padStart(2, "0");
        hudTimestamp.textContent = `${yy}/${mm}/${dd} ${hh}:${mi}:${ss} UTC`;
    }
    setInterval(updateHudClock, 1000);
    updateHudClock();

    // 3. Ayarları Sunucuya Gönderme
    async function pushSettings(payload) {
        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                const data = await res.json();
                syncUIFromState(data);
            }
        } catch (err) {
            console.error("Ayarlar güncellenemedi:", err);
        }
    }

    function scheduleThresholdSync() {
        if (settingsDebounce) clearTimeout(settingsDebounce);
        settingsDebounce = setTimeout(() => {
            pushSettings({
                conf: Number(sliderEsik1.value),
                iou: Number(sliderEsik2.value),
                alarm: Number(sliderEsik3.value),
            });
        }, 180);
    }

    sliderEsik1.addEventListener("input", () => {
        valEsik1.textContent = `${sliderEsik1.value}%`;
        scheduleThresholdSync();
    });
    sliderEsik2.addEventListener("input", () => {
        valEsik2.textContent = `${sliderEsik2.value}%`;
        scheduleThresholdSync();
    });
    sliderEsik3.addEventListener("input", () => {
        valEsik3.textContent = `${sliderEsik3.value}%`;
        scheduleThresholdSync();
    });

    // 4. Canlı Akış / Derin Tarama Anahtarı
    modeSwitchBtn.addEventListener("click", () => {
        isDeepScan = !isDeepScan;
        modeSwitchBtn.setAttribute("aria-checked", String(isDeepScan));
        hudBadgeText.textContent = isDeepScan ? "DERİN TARAMA" : "CANLI AKIŞ";
        pushSettings({ deep_scan: isDeepScan });
    });

    // 5. Başla / Durdur Butonu
    startBtn.addEventListener("click", () => {
        isRunning = !isRunning;
        pushSettings({ running: isRunning });
    });

    // 6. Dosya Sürükle & Bırak ve Yükleme
    async function handleFileUpload(file) {
        if (!file) return;
        uploadText.textContent = `Yükleniyor: ${file.name}...`;
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            if (res.ok) {
                const info = await res.json();
                uploadText.textContent = `Yüklendi: ${info.filename}`;
                if (info.type === "metrics") {
                    loadTrainingMetrics();
                }
                fetchState();
            } else {
                uploadText.textContent = "Dosya yükleme hatası!";
            }
        } catch (err) {
            uploadText.textContent = "Bağlantı hatası!";
        }
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // 7. Durum, Sistem Günlükleri ve Bulunan Eşyalar Güncellemesi
    function syncUIFromState(data) {
        isRunning = Boolean(data.running);
        startBtn.textContent = isRunning ? "Durdur" : "Başla";
        startBtn.classList.toggle("running", isRunning);

        if (data.alarm_active) {
            alarmBanner.classList.remove("hidden");
        } else {
            alarmBanner.classList.add("hidden");
        }

        // Sistem Günlükleri
        if (Array.isArray(data.logs)) {
            logsContainer.innerHTML = data.logs
                .map((l) => {
                    const cls = l.level === "alarm" ? "log-entry alarm" : "log-entry";
                    return `<div class="${cls}">[${l.time}] ${l.message}</div>`;
                })
                .join("");
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        // Bulunan Eşyalar
        if (Array.isArray(data.found_items)) {
            foundItemsList.innerHTML = data.found_items
                .map((item) => {
                    const thumbSrc = item.thumbnail || "";
                    const confPct = Math.round((item.confidence || 0) * 100);
                    return `
                        <div class="found-item" title="Güven: %${confPct} | Risk: ${item.risk || 'Orta'}">
                            <img class="found-thumb" src="${thumbSrc}" alt="${item.tr_label}" />
                            <div class="found-info">
                                <span class="found-title">${item.tr_label}</span>
                                <span class="found-sector">${item.sector}</span>
                            </div>
                        </div>
                    `;
                })
                .join("");
        }
    }

    async function fetchState() {
        try {
            const res = await fetch("/api/state");
            if (res.ok) {
                const data = await res.json();
                syncUIFromState(data);
            }
        } catch (_) {
            // Sunucu meşgulse sessizce geç
        }
    }

    setInterval(fetchState, 1000);
    fetchState();

    // 8. Eğitim Grafikleri Çizim Motoru (Figma 0-1200 Y-Ekseni + İkili Eğri)
    function drawDualLineChart(canvas, blueSeries, greySeries) {
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(300, rect.width);
        const height = Math.max(220, rect.height);

        canvas.width = width * dpr;
        canvas.height = height * dpr;

        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);

        const isDark = htmlEl.getAttribute("data-theme") === "dark";
        const gridColor = isDark ? "rgba(226, 232, 240, 0.72)" : "#e6ecf3";
        const labelColor = isDark ? "#8b9bb0" : "#7a869a";

        const padLeft = 42;
        const padRight = 12;
        const padTop = 12;
        const padBottom = 14;
        const plotW = width - padLeft - padRight;
        const plotH = height - padTop - padBottom;

        const yMax = 1200;
        const yStep = 100;

        ctx.font = '10.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        // Yatay ızgara çizgileri ve Y ekseni etiketleri (0..1200)
        for (let val = 0; val <= yMax; val += yStep) {
            const ratio = val / yMax;
            const y = padTop + plotH - ratio * plotH;

            ctx.fillStyle = labelColor;
            ctx.fillText(String(val), padLeft - 8, y);

            ctx.beginPath();
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.moveTo(padLeft, y);
            ctx.lineTo(width - padRight, y);
            ctx.stroke();
        }

        function plotSeries(series, strokeColor, lineWidth) {
            if (!Array.isArray(series) || series.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = lineWidth;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";

            const n = series.length;
            for (let i = 0; i < n; i++) {
                const x = padLeft + (i / (n - 1)) * plotW;
                const clamped = Math.max(0, Math.min(yMax, Number(series[i]) || 0));
                const y = padTop + plotH - (clamped / yMax) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Önce gri seri, üstüne mavi seri (Figma tasarımındaki katman sırası)
        plotSeries(greySeries, "#78879b", 1.35);
        plotSeries(blueSeries, "#1565c0", 1.6);
    }

    function renderBothCharts(metrics) {
        if (!metrics) return;
        drawDualLineChart(chartLeftCanvas, metrics.chart1.blue, metrics.chart1.grey);
        drawDualLineChart(chartRightCanvas, metrics.chart2.blue, metrics.chart2.grey);
    }

    async function loadTrainingMetrics() {
        try {
            const res = await fetch("/api/training-metrics");
            if (res.ok) {
                cachedMetrics = await res.json();
                renderBothCharts(cachedMetrics);
            }
        } catch (err) {
            console.error("Eğitim grafikleri yüklenemedi:", err);
        }
    }

    window.addEventListener("resize", () => {
        if (cachedMetrics) renderBothCharts(cachedMetrics);
    });

    loadTrainingMetrics();
})();
