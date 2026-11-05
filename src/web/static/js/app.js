/**
 * Sezgen — Havalimanı Pist FOD Tespit Sistemi
 * Gerçek Video Oynatıcı, YOLOv26 Model Yöneticisi, Canlı Tespit ve Gerçek Eğitim Grafikleri İstemcisi
 */

(() => {
    const htmlEl = document.documentElement;
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeToggleLabel = document.getElementById("themeToggleLabel");
    const deviceBadge = document.getElementById("deviceBadge");

    // Sol Panel: Video & Medya
    const mediaDropZone = document.getElementById("mediaDropZone");
    const mediaFileInput = document.getElementById("mediaFileInput");
    const mediaUploadText = document.getElementById("mediaUploadText");
    const mediaProgressWrap = document.getElementById("mediaProgressWrap");
    const mediaProgressBar = document.getElementById("mediaProgressBar");
    const existingMediaSelect = document.getElementById("existingMediaSelect");
    const rtspInput = document.getElementById("rtspInput");
    const rtspConnectBtn = document.getElementById("rtspConnectBtn");

    // Sol Panel: YOLOv26 Model Ayarları
    const refreshModelsBtn = document.getElementById("refreshModelsBtn");
    const modelStatusBox = document.getElementById("modelStatusBox");
    const modelStatusTitle = document.getElementById("modelStatusTitle");
    const modelStatusMeta = document.getElementById("modelStatusMeta");
    const modelSelect = document.getElementById("modelSelect");
    const modelFileInput = document.getElementById("modelFileInput");
    const modelUploadBtnText = document.getElementById("modelUploadBtnText");
    const modelProgressWrap = document.getElementById("modelProgressWrap");
    const modelProgressBar = document.getElementById("modelProgressBar");
    const imgszSelect = document.getElementById("imgszSelect");

    // Mod ve Kaydırıcılar
    const modeSwitchBtn = document.getElementById("modeSwitchBtn");
    const sliderConf = document.getElementById("sliderConf");
    const sliderIou = document.getElementById("sliderIou");
    const sliderAlarm = document.getElementById("sliderAlarm");
    const valConf = document.getElementById("valConf");
    const valIou = document.getElementById("valIou");
    const valAlarm = document.getElementById("valAlarm");
    const loopVideoCheck = document.getElementById("loopVideoCheck");

    // Oynatma & Analiz Butonları
    const startBtn = document.getElementById("startBtn");
    const restartBtn = document.getElementById("restartBtn");
    const stepBtn = document.getElementById("stepBtn");

    // Orta Panel: Video Görüntüsü & Kontroller
    const videoCard = document.getElementById("videoCard");
    const videoEmptyState = document.getElementById("videoEmptyState");
    const emptyStateModelHint = document.getElementById("emptyStateModelHint");
    const videoDisplayImg = document.getElementById("videoDisplayImg");
    const hudOverlay = document.getElementById("hudOverlay");
    const hudSourceLine = document.getElementById("hudSourceLine");
    const hudMetaLine = document.getElementById("hudMetaLine");
    const hudStatusDot = document.getElementById("hudStatusDot");
    const hudModeText = document.getElementById("hudModeText");
    const hudModelBadge = document.getElementById("hudModelBadge");
    const alarmBanner = document.getElementById("alarmBanner");
    const alarmBannerText = document.getElementById("alarmBannerText");

    const videoTransportBar = document.getElementById("videoTransportBar");
    const transportPlayBtn = document.getElementById("transportPlayBtn");
    const seekSlider = document.getElementById("seekSlider");
    const transportTimeLabel = document.getElementById("transportTimeLabel");
    const fullscreenBtn = document.getElementById("fullscreenBtn");

    // Sistem Günlükleri & Bulunan Eşyalar
    const logsContainer = document.getElementById("logsContainer");
    const clearLogsBtn = document.getElementById("clearLogsBtn");
    const foundCountBadge = document.getElementById("foundCountBadge");
    const foundEmptyState = document.getElementById("foundEmptyState");
    const foundItemsList = document.getElementById("foundItemsList");
    const clearDetectionsBtn = document.getElementById("clearDetectionsBtn");

    // Eğitim Grafikleri
    const metricsSourceBadge = document.getElementById("metricsSourceBadge");
    const metricsCsvInput = document.getElementById("metricsCsvInput");
    const metricsSummaryStrip = document.getElementById("metricsSummaryStrip");
    const chartsEmptyState = document.getElementById("chartsEmptyState");
    const chartsGrid = document.getElementById("chartsGrid");
    const leftChartTitle = document.getElementById("leftChartTitle");
    const rightChartTitle = document.getElementById("rightChartTitle");
    const leftChartLegend = document.getElementById("leftChartLegend");
    const rightChartLegend = document.getElementById("rightChartLegend");
    const chartLeftCanvas = document.getElementById("trainingChartLeft");
    const chartRightCanvas = document.getElementById("trainingChartRight");
    const tooltipLeft = document.getElementById("tooltipLeft");
    const tooltipRight = document.getElementById("tooltipRight");

    let appState = null;
    let isStreamingActive = false;
    let lastRenderedSeq = -1;
    let lastLogSignature = "";
    let lastFoundSignature = "";
    let isSeekingSlider = false;
    let settingsTimer = null;
    let cachedMetrics = null;

    // 1. Tema Yönetimi
    function applyTheme(theme) {
        htmlEl.setAttribute("data-theme", theme);
        localStorage.setItem("sezgen_theme", theme);
        themeToggleLabel.textContent = theme === "dark" ? "Karanlık tema" : "Aydınlık tema";
        if (cachedMetrics && cachedMetrics.available) {
            renderTrainingCharts(cachedMetrics);
        }
    }

    applyTheme(localStorage.getItem("sezgen_theme") || "light");

    themeToggleBtn.addEventListener("click", () => {
        const cur = htmlEl.getAttribute("data-theme") === "dark" ? "dark" : "light";
        applyTheme(cur === "dark" ? "light" : "dark");
    });

    // 2. Yardımcı API İstek Fonksiyonu
    async function apiPost(url, bodyObj = null) {
        const opts = { method: "POST" };
        if (bodyObj !== null) {
            opts.headers = { "Content-Type": "application/json" };
            opts.body = JSON.stringify(bodyObj);
        }
        const res = await fetch(url, opts);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            alert(data.detail || "İşlem sırasında bir hata oluştu.");
            throw new Error(data.detail || "API Hatası");
        }
        return data;
    }

    // İlerleme çubuklu gerçek dosya yükleme (XHR)
    function uploadWithProgress(url, file, progressWrap, progressBar, onDone) {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append("file", file);

        if (progressWrap && progressBar) {
            progressWrap.classList.remove("hidden");
            progressBar.style.width = "0%";
        }

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && progressBar) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = `${pct}%`;
            }
        };

        xhr.onload = () => {
            if (progressWrap) {
                setTimeout(() => progressWrap.classList.add("hidden"), 500);
            }
            let resp = {};
            try {
                resp = JSON.parse(xhr.responseText);
            } catch (_) {}

            if (xhr.status >= 200 && xhr.status < 300) {
                onDone(null, resp);
            } else {
                onDone(new Error(resp.detail || "Dosya yüklenemedi."), null);
            }
        };

        xhr.onerror = () => {
            if (progressWrap) progressWrap.classList.add("hidden");
            onDone(new Error("Ağ bağlantı hatası."), null);
        };

        xhr.open("POST", url, true);
        xhr.send(formData);
    }

    // 3. Medya (Video / Görsel) Yükleme & Seçme
    function triggerMediaUpload(file) {
        if (!file) return;
        mediaUploadText.textContent = `Yükleniyor: ${file.name}...`;
        uploadWithProgress("/api/upload/media", file, mediaProgressWrap, mediaProgressBar, (err, stateData) => {
            if (err) {
                mediaUploadText.textContent = `Hata: ${err.message}`;
                alert(err.message);
                return;
            }
            mediaUploadText.textContent = `Aktif Dosya: ${file.name}`;
            lastRenderedSeq = -1;
            syncUI(stateData);
        });
    }

    mediaFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            triggerMediaUpload(e.target.files[0]);
            e.target.value = "";
        }
    });

    mediaDropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        mediaDropZone.classList.add("dragover");
    });
    mediaDropZone.addEventListener("dragleave", () => {
        mediaDropZone.classList.remove("dragover");
    });
    mediaDropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        mediaDropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const f = e.dataTransfer.files[0];
            const lower = f.name.toLowerCase();
            if (lower.endsWith(".pt") || lower.endsWith(".onnx")) {
                triggerModelUpload(f);
            } else if (lower.endsWith(".csv")) {
                triggerMetricsUpload(f);
            } else {
                triggerMediaUpload(f);
            }
        }
    });

    existingMediaSelect.addEventListener("change", async () => {
        const val = existingMediaSelect.value;
        if (!val) return;
        try {
            const st = await apiPost("/api/media/select", { filename: val });
            mediaUploadText.textContent = `Aktif Dosya: ${val}`;
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    rtspConnectBtn.addEventListener("click", async () => {
        const url = rtspInput.value.trim();
        if (!url) {
            alert("Lütfen bir RTSP adresi (rtsp://...) veya kamera numarası (0) girin.");
            return;
        }
        try {
            const st = await apiPost("/api/media/rtsp", { url });
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    // 4. YOLOv26 Model Yükleme, Seçme ve Yenileme
    function triggerModelUpload(file) {
        if (!file) return;
        modelUploadBtnText.textContent = "Yükleniyor...";
        uploadWithProgress("/api/upload/model", file, modelProgressWrap, modelProgressBar, (err, stateData) => {
            modelUploadBtnText.textContent = "best.pt Yükle";
            if (err) {
                alert(err.message);
                return;
            }
            lastRenderedSeq = -1;
            syncUI(stateData);
            // Model içinde gömülü eğitim geçmişi varsa grafikleri otomatik çiz!
            loadTrainingMetrics();
        });
    }

    modelFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            triggerModelUpload(e.target.files[0]);
            e.target.value = "";
        }
    });

    modelSelect.addEventListener("change", async () => {
        const val = modelSelect.value;
        if (!val) return;
        try {
            const st = await apiPost("/api/model/select", { model_name: val });
            lastRenderedSeq = -1;
            syncUI(st);
            loadTrainingMetrics();
        } catch (_) {}
    });

    refreshModelsBtn.addEventListener("click", async () => {
        try {
            const st = await apiPost("/api/model/refresh");
            syncUI(st);
            loadTrainingMetrics();
        } catch (_) {}
    });

    imgszSelect.addEventListener("change", async () => {
        try {
            const st = await apiPost("/api/settings", { imgsz: Number(imgszSelect.value) });
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    // 5. Eşik Kaydırıcıları & Tarama Modu
    function scheduleSettingsPush() {
        if (settingsTimer) clearTimeout(settingsTimer);
        settingsTimer = setTimeout(async () => {
            try {
                const st = await apiPost("/api/settings", {
                    conf: Number(sliderConf.value),
                    iou: Number(sliderIou.value),
                    alarm: Number(sliderAlarm.value),
                });
                lastRenderedSeq = -1;
                syncUI(st);
            } catch (_) {}
        }, 180);
    }

    sliderConf.addEventListener("input", () => {
        valConf.textContent = `${sliderConf.value}%`;
        scheduleSettingsPush();
    });
    sliderIou.addEventListener("input", () => {
        valIou.textContent = `${sliderIou.value}%`;
        scheduleSettingsPush();
    });
    sliderAlarm.addEventListener("input", () => {
        valAlarm.textContent = `${sliderAlarm.value}%`;
        scheduleSettingsPush();
    });

    modeSwitchBtn.addEventListener("click", async () => {
        if (modeSwitchBtn.disabled) return;
        const cur = modeSwitchBtn.getAttribute("aria-checked") === "true";
        const next = !cur;
        modeSwitchBtn.setAttribute("aria-checked", String(next));
        try {
            const st = await apiPost("/api/settings", { deep_scan: next });
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    loopVideoCheck.addEventListener("change", async () => {
        try {
            const st = await apiPost("/api/settings", { loop_video: loopVideoCheck.checked });
            syncUI(st);
        } catch (_) {}
    });

    // 6. Video Oynatma, Duraklatma, Başa Sarma, +1 Kare ve Konumlandırma (Seek)
    async function togglePlayPause() {
        try {
            const st = await apiPost("/api/control/toggle");
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    }

    startBtn.addEventListener("click", togglePlayPause);
    transportPlayBtn.addEventListener("click", togglePlayPause);

    restartBtn.addEventListener("click", async () => {
        try {
            const st = await apiPost("/api/control/restart");
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    stepBtn.addEventListener("click", async () => {
        try {
            const st = await apiPost("/api/control/step");
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    seekSlider.addEventListener("mousedown", () => { isSeekingSlider = true; });
    seekSlider.addEventListener("touchstart", () => { isSeekingSlider = true; });
    seekSlider.addEventListener("change", async () => {
        const targetFrame = Number(seekSlider.value);
        isSeekingSlider = false;
        try {
            const st = await apiPost("/api/control/seek", { frame_idx: targetFrame });
            lastRenderedSeq = -1;
            syncUI(st);
        } catch (_) {}
    });

    fullscreenBtn.addEventListener("click", () => {
        if (!document.fullscreenElement) {
            videoCard.requestFullscreen().catch(() => {});
        } else {
            document.exitFullscreen().catch(() => {});
        }
    });

    // 7. Günlükleri ve Tespitleri Temizleme
    clearLogsBtn.addEventListener("click", async () => {
        try {
            const st = await apiPost("/api/logs/clear");
            syncUI(st);
        } catch (_) {}
    });

    clearDetectionsBtn.addEventListener("click", async () => {
        try {
            const st = await apiPost("/api/detections/clear");
            syncUI(st);
        } catch (_) {}
    });

    // 8. Arayüz Durum Senkronizasyonu
    function syncUI(data) {
        if (!data) return;
        appState = data;

        const { running, loop_video, media, model, settings, telemetry, logs, found_items } = data;

        // Üst Cihaz Rozeti
        deviceBadge.textContent = model.device || "CPU";

        // Model Durum Kutusu
        if (model.loaded) {
            modelStatusBox.classList.remove("waiting");
            modelStatusBox.classList.add("ready");
            modelStatusTitle.textContent = `Aktif: ${model.model_name}`;
            modelStatusMeta.textContent = `${model.num_classes} Sınıf • imgsz ${settings.imgsz} • ${model.device}`;
            emptyStateModelHint.textContent = `Aktif Model: ${model.model_name} (${model.num_classes} sınıf hazır)`;
        } else {
            modelStatusBox.classList.remove("ready");
            modelStatusBox.classList.add("waiting");
            modelStatusTitle.textContent = "Model Yüklenmedi";
            modelStatusMeta.textContent = model.error || "Analiz için best.pt (.pt / .onnx) yükleyin";
            emptyStateModelHint.textContent = "Model Bekleniyor: Sol paneldeki Model Ayarları'ndan best.pt yükleyin";
        }

        // Model Seçici Dropdown
        const availModels = model.available_models || [];
        const modelSig = availModels.map((m) => m.name).join("|") + "::" + (model.model_name || "");
        if (modelSelect.dataset.sig !== modelSig) {
            modelSelect.dataset.sig = modelSig;
            if (availModels.length === 0) {
                modelSelect.innerHTML = `<option value="">— models/ içinde model yok —</option>`;
            } else {
                modelSelect.innerHTML = availModels
                    .map((m) => {
                        const sel = m.name === model.model_name ? "selected" : "";
                        return `<option value="${m.name}" ${sel}>${m.name} (${m.size_mb} MB)</option>`;
                    })
                    .join("");
            }
        }

        // Yüklü Medya Seçici Dropdown
        const uploadedFiles = media.uploaded_files || [];
        const mediaSig = uploadedFiles.map((f) => f.name).join("|") + "::" + (media.source_name || "");
        if (existingMediaSelect.dataset.sig !== mediaSig) {
            existingMediaSelect.dataset.sig = mediaSig;
            let opts = `<option value="">— Yüklü Medya Seçin (${uploadedFiles.length}) —</option>`;
            opts += uploadedFiles
                .map((f) => {
                    const sel = f.name === media.source_name ? "selected" : "";
                    return `<option value="${f.name}" ${sel}>[${f.type.toUpperCase()}] ${f.name} (${f.size_mb} MB)</option>`;
                })
                .join("");
            existingMediaSelect.innerHTML = opts;
        }

        // Ayarlar Senkronizasyonu
        if (document.activeElement !== imgszSelect) {
            imgszSelect.value = String(settings.imgsz);
        }
        modeSwitchBtn.setAttribute("aria-checked", String(Boolean(settings.deep_scan)));
        const isImage = media.type === "image";
        const isVideo = media.type === "video";
        const liveAllowed = settings.live_allowed !== false && !isImage;
        const videoEnded =
            isVideo && !running && media.total_frames > 0 && media.current_frame >= media.total_frames - 1;

        // Görsel: anahti kilitle, "Derin Tarama" zorunlu, "Canlı Akış" karart
        modeSwitchBtn.disabled = isImage;
        modeSwitchBtn.classList.toggle("is-disabled", isImage);
        modeSwitchBtn.title = isImage
            ? "Görsel girdide yalnızca Derin Tarama kullanılabilir"
            : "Canlı Akış (Tam Kare YOLO) / Derin Tarama (SAHI)";
        document.getElementById("labelLiveStream").classList.toggle("is-disabled", !liveAllowed);
        document.getElementById("labelDeepScan").classList.toggle("is-forced", isImage);

        // Başa Sar: görselde devre dışı + karartılmış; videoda yalnızca oynatma bitince aktif
        restartBtn.disabled = isImage || !videoEnded;
        restartBtn.classList.toggle("is-disabled", restartBtn.disabled);
        restartBtn.title = isImage
            ? "Görselde kullanılamaz"
            : videoEnded
              ? "Videoyu Başa Sar"
              : "Video bitince aktif olur";
        loopVideoCheck.checked = Boolean(loop_video);

        // Başla / Duraklat Butonları
        if (media.type === "image") {
            startBtn.textContent = "Yeniden Analiz Et";
            startBtn.classList.remove("running");
        } else {
            startBtn.textContent = running ? "Duraklat" : "Başla";
            startBtn.classList.toggle("running", running);
        }
        transportPlayBtn.textContent = running ? "⏸" : "▶";

        // Video Ekranı & Akış Kaynağı Yönetimi
        if (!media.loaded) {
            videoEmptyState.classList.remove("hidden");
            videoDisplayImg.classList.add("hidden");
            hudOverlay.classList.add("hidden");
            videoTransportBar.classList.add("hidden");
            if (isStreamingActive) {
                videoDisplayImg.removeAttribute("src");
                isStreamingActive = false;
            }
        } else {
            videoEmptyState.classList.add("hidden");
            videoDisplayImg.classList.remove("hidden");
            hudOverlay.classList.remove("hidden");

            if (media.type === "video") {
                videoTransportBar.classList.remove("hidden");
                seekSlider.max = Math.max(1, media.total_frames - 1);
                if (!isSeekingSlider) {
                    seekSlider.value = media.current_frame;
                }
                transportTimeLabel.textContent = `${media.current_time} / ${media.duration_str} (Kare ${media.current_frame}/${media.total_frames})`;
            } else {
                videoTransportBar.classList.add("hidden");
            }

            // Canlı oynatmada MJPEG stream, duraklatıldığında/görselde tekil net snapshot
            if (running && (media.type === "video" || media.type === "stream")) {
                if (!isStreamingActive) {
                    videoDisplayImg.src = `/api/stream?ts=${Date.now()}`;
                    isStreamingActive = true;
                }
            } else {
                if (isStreamingActive || media.frame_seq !== lastRenderedSeq) {
                    isStreamingActive = false;
                    lastRenderedSeq = media.frame_seq;
                    videoDisplayImg.src = `/api/snapshot?seq=${media.frame_seq}&ts=${Date.now()}`;
                }
            }

            // HUD Bilgileri
            hudSourceLine.textContent = `${media.source_name} (${media.width}x${media.height})`;
            const fpsTxt = telemetry.processing_fps > 0 ? `${telemetry.processing_fps} FPS` : `${media.native_fps} FPS`;
            hudMetaLine.textContent =
                media.type === "video"
                    ? `Kare ${media.current_frame}/${media.total_frames} • ${fpsTxt}`
                    : `${media.type.toUpperCase()} • ${fpsTxt}`;

            hudStatusDot.classList.toggle("live", running);
            const modeLabel = settings.deep_scan ? "DERİN TARAMA (SAHI)" : "CANLI AKIŞ";
            hudModeText.textContent = running ? modeLabel : `DURAKLATILDI • ${modeLabel}`;

            if (model.loaded) {
                hudModelBadge.textContent = `MODEL: ${model.model_name}`;
                hudModelBadge.className = "hud-badge model-chip active";
            } else {
                hudModelBadge.textContent = "MODELSİZ ÖNİZLEME";
                hudModelBadge.className = "hud-badge model-chip missing";
            }
        }

        // Gerçek FOD Alarm Göstergesi
        if (telemetry.alarm_active) {
            alarmBanner.classList.remove("hidden");
            alarmBannerText.textContent = `FOD ALARMI — BU KAREDE ${telemetry.current_frame_detections} CİSİM TESPİT EDİLDİ!`;
        } else {
            alarmBanner.classList.add("hidden");
        }

        // Gerçek Sistem Günlükleri (Yalnızca değişiklik olduğunda DOM güncelle)
        if (Array.isArray(logs)) {
            const sig = `${logs.length}_${logs.length ? logs[logs.length - 1].time + logs[logs.length - 1].message : ""}`;
            if (sig !== lastLogSignature) {
                lastLogSignature = sig;
                logsContainer.innerHTML = logs
                    .map((l) => `<div class="log-entry ${l.level || "info"}">[${l.time}] ${l.message}</div>`)
                    .join("");
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }
        }

        // Gerçek Bulunan Eşyalar
        foundCountBadge.textContent = String(telemetry.total_detections || 0);
        if (Array.isArray(found_items)) {
            if (found_items.length === 0) {
                foundEmptyState.classList.remove("hidden");
                foundItemsList.innerHTML = "";
                lastFoundSignature = "empty";
            } else {
                foundEmptyState.classList.add("hidden");
                const fSig = found_items.map((d) => d.id).join("|");
                if (fSig !== lastFoundSignature) {
                    lastFoundSignature = fSig;
                    foundItemsList.innerHTML = found_items
                        .map((item) => {
                            const pct = (Number(item.confidence || 0) * 100).toFixed(1);
                            const riskCls =
                                item.risk === "Yüksek" ? "high" : item.risk === "Orta" ? "med" : "low";
                            return `
                                <div class="found-item" data-frame="${item.frame_idx}" title="Videoda Kare #${item.frame_idx} konumuna gitmek için tıklayın | Koordinat: [${item.bbox}]">
                                    <img class="found-thumb" src="${item.thumbnail}" alt="${item.tr_label}" />
                                    <div class="found-info">
                                        <div class="found-top-row">
                                            <span class="found-title">${item.tr_label}</span>
                                            <span class="risk-pill ${riskCls}">${item.risk}</span>
                                        </div>
                                        <span class="found-meta">%${pct} • ${item.class_name}</span>
                                        <span class="found-meta">${item.video_time} • Kare #${item.frame_idx}</span>
                                    </div>
                                </div>
                            `;
                        })
                        .join("");

                    // Bulunan eşyaya tıklandığında videoyu doğrudan o kareye sar!
                    foundItemsList.querySelectorAll(".found-item").forEach((el) => {
                        el.addEventListener("click", async () => {
                            const fIdx = Number(el.dataset.frame || 0);
                            if (appState && appState.media.type === "video" && fIdx > 0) {
                                try {
                                    // Tam tespit edilen kareyi okumak için fIdx - 1 konumuna seek yapıp 1 kare okutuyoruz
                                    const st = await apiPost("/api/control/seek", { frame_idx: Math.max(0, fIdx - 1) });
                                    lastRenderedSeq = -1;
                                    syncUI(st);
                                } catch (_) {}
                            }
                        });
                    });
                }
            }
        }
    }

    async function pollState() {
        try {
            const res = await fetch("/api/state");
            if (res.ok) {
                const st = await res.json();
                syncUI(st);
            }
        } catch (_) {}
    }

    setInterval(pollState, 650);
    pollState();

    // 9. Gerçek Eğitim Grafikleri (results.csv veya best.pt İçi Kayıt)
    function triggerMetricsUpload(file) {
        if (!file) return;
        uploadWithProgress("/api/upload/metrics", file, null, null, (err, metricsData) => {
            if (err) {
                alert(err.message);
                return;
            }
            cachedMetrics = metricsData;
            renderTrainingSection(metricsData);
            pollState();
        });
    }

    metricsCsvInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            triggerMetricsUpload(e.target.files[0]);
            e.target.value = "";
        }
    });

    function buildLegendHTML(seriesList) {
        return (seriesList || [])
            .map(
                (s) => `
                <span class="legend-item">
                    <span class="legend-swatch" style="background:${s.color}"></span>
                    <span>${s.name}</span>
                </span>
            `
            )
            .join("");
    }

    function drawRealMultiSeriesChart(canvas, tooltipEl, epochs, chartConfig) {
        if (!canvas || !chartConfig || !Array.isArray(chartConfig.series) || chartConfig.series.length === 0) {
            return;
        }

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
        const gridColor = isDark ? "rgba(148, 163, 184, 0.22)" : "#e3ebf3";
        const labelColor = isDark ? "#94a3b8" : "#64748b";

        const allVals = [];
        chartConfig.series.forEach((s) => s.data.forEach((v) => { if (Number.isFinite(v)) allVals.push(v); }));

        let yMin = typeof chartConfig.y_min === "number" ? chartConfig.y_min : 0;
        let yMax = typeof chartConfig.y_max === "number" ? chartConfig.y_max : Math.max(...allVals, 0.1) * 1.1;
        if (yMax <= yMin) yMax = yMin + 1;

        const padLeft = 48;
        const padRight = 14;
        const padTop = 14;
        const padBottom = 26;
        const plotW = width - padLeft - padRight;
        const plotH = height - padTop - padBottom;

        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        const yTicks = 6;
        for (let i = 0; i <= yTicks; i++) {
            const ratio = i / yTicks;
            const val = yMin + ratio * (yMax - yMin);
            const y = padTop + plotH - ratio * plotH;

            const formatted = yMax <= 10 ? val.toFixed(2) : `${Math.round(val)}${chartConfig.unit || ""}`;
            ctx.fillStyle = labelColor;
            ctx.fillText(formatted, padLeft - 8, y);

            ctx.beginPath();
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.moveTo(padLeft, y);
            ctx.lineTo(width - padRight, y);
            ctx.stroke();
        }

        // X Ekseni Epoch Etiketleri
        const nPoints = epochs.length;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const xTicks = Math.min(8, Math.max(2, nPoints));
        for (let t = 0; t < xTicks; t++) {
            const idx = nPoints === 1 ? 0 : Math.round((t / (xTicks - 1)) * (nPoints - 1));
            const x = nPoints === 1 ? padLeft + plotW / 2 : padLeft + (idx / (nPoints - 1)) * plotW;
            ctx.fillStyle = labelColor;
            ctx.fillText(`Ep ${epochs[idx]}`, x, padTop + plotH + 7);
        }

        // Serileri Çiz
        chartConfig.series.forEach((s) => {
            if (!Array.isArray(s.data) || s.data.length === 0) return;
            ctx.beginPath();
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";

            const len = s.data.length;
            for (let i = 0; i < len; i++) {
                const x = len === 1 ? padLeft + plotW / 2 : padLeft + (i / (len - 1)) * plotW;
                const clamped = Math.max(yMin, Math.min(yMax, Number(s.data[i]) || 0));
                const y = padTop + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            if (len === 1) {
                const y = padTop + plotH - ((s.data[0] - yMin) / (yMax - yMin)) * plotH;
                ctx.fillStyle = s.color;
                ctx.beginPath();
                ctx.arc(padLeft + plotW / 2, y, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        // İnteraktif Hover Tooltip
        canvas.onmousemove = (ev) => {
            if (!tooltipEl || nPoints === 0) return;
            const r = canvas.getBoundingClientRect();
            const mx = ev.clientX - r.left;
            if (mx < padLeft || mx > width - padRight) {
                tooltipEl.classList.add("hidden");
                return;
            }
            const rel = (mx - padLeft) / Math.max(1, plotW);
            const idx = Math.max(0, Math.min(nPoints - 1, Math.round(rel * (nPoints - 1))));
            let html = `<strong>Epoch ${epochs[idx]}</strong><br/>`;
            chartConfig.series.forEach((s) => {
                if (idx < s.data.length) {
                    const v = s.data[idx];
                    const disp = yMax <= 10 ? Number(v).toFixed(4) : `${Number(v).toFixed(2)}${chartConfig.unit || ""}`;
                    html += `<span style="color:${s.color}">●</span> ${s.name}: <strong>${disp}</strong><br/>`;
                }
            });
            tooltipEl.innerHTML = html;
            tooltipEl.classList.remove("hidden");
            const leftPos = Math.min(width - 175, Math.max(8, mx + 12));
            tooltipEl.style.left = `${leftPos}px`;
            tooltipEl.style.top = `14px`;
        };

        canvas.onmouseleave = () => {
            if (tooltipEl) tooltipEl.classList.add("hidden");
        };
    }

    function renderTrainingCharts(m) {
        leftChartTitle.textContent = m.loss_chart.title;
        rightChartTitle.textContent = m.metrics_chart.title;
        leftChartLegend.innerHTML = buildLegendHTML(m.loss_chart.series);
        rightChartLegend.innerHTML = buildLegendHTML(m.metrics_chart.series);

        drawRealMultiSeriesChart(chartLeftCanvas, tooltipLeft, m.epochs, m.loss_chart);
        drawRealMultiSeriesChart(chartRightCanvas, tooltipRight, m.epochs, m.metrics_chart);
    }

    function renderTrainingSection(m) {
        if (!m || !m.available) {
            metricsSourceBadge.textContent = "Veri Bekleniyor";
            metricsSummaryStrip.classList.add("hidden");
            chartsGrid.classList.add("hidden");
            chartsEmptyState.classList.remove("hidden");
            return;
        }

        metricsSourceBadge.textContent = `Kaynak: ${m.source}`;
        chartsEmptyState.classList.add("hidden");
        chartsGrid.classList.remove("hidden");

        // Özet KPI Kartları
        const s = m.summary || {};
        const kpis = [
            { label: "Toplam Epoch", val: s.total_epochs ?? "—" },
            { label: "En İyi mAP@0.5", val: s.best_map50 != null ? `%${s.best_map50}` : "—" },
            { label: "En İyi mAP@0.5:0.95", val: s.best_map50_95 != null ? `%${s.best_map50_95}` : "—" },
            { label: "En İyi Recall", val: s.best_recall != null ? `%${s.best_recall}` : "—" },
            { label: "En İyi Precision", val: s.best_precision != null ? `%${s.best_precision}` : "—" },
            { label: "Min Box Loss", val: s.min_val_box_loss != null ? s.min_val_box_loss : "—" },
        ];
        metricsSummaryStrip.innerHTML = kpis
            .map(
                (k) => `
                <div class="kpi-box">
                    <span class="kpi-label">${k.label}</span>
                    <span class="kpi-value">${k.val}</span>
                </div>
            `
            )
            .join("");
        metricsSummaryStrip.classList.remove("hidden");

        requestAnimationFrame(() => renderTrainingCharts(m));
    }

    async function loadTrainingMetrics() {
        try {
            const res = await fetch("/api/training-metrics");
            if (res.ok) {
                cachedMetrics = await res.json();
                renderTrainingSection(cachedMetrics);
            }
        } catch (_) {}
    }

    window.addEventListener("resize", () => {
        if (cachedMetrics && cachedMetrics.available) {
            renderTrainingCharts(cachedMetrics);
        }
    });

    loadTrainingMetrics();
})();
