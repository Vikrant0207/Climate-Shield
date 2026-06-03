const API_URL =
    window.location.hostname === "127.0.0.1"
    ||
    window.location.hostname === "localhost"
        ? "http://127.0.0.1:5000/weather"
        : window.location.origin + "/weather";

// 🌐 1. Create a global variable to hold our map instance
let climateMap = null;

async function getWeatherData() {
    const city = document.getElementById('city').value.trim();
    const state = document.getElementById('state').value.trim();
    const country = document.getElementById('country').value.trim();
    const loading = document.getElementById('loading');
    const messageBox = document.getElementById('message-box');
    const results = document.getElementById('results');
    const alertBox = document.getElementById('alert-box');
    const resultStatus = document.getElementById('result-status');
    const resultSummary = document.getElementById('result-summary');
    const statusPill = document.getElementById('status-pill');

    const showMessage = (message, tone) => {
        messageBox.textContent = message;
        messageBox.classList.remove(
            'hidden',
            'is-error',
            'is-success'
        );
        if (tone) {
            messageBox.classList.add(tone);
        }
    };

    const hideMessage = () => {
        messageBox.textContent = '';
        messageBox.classList.add('hidden');
        messageBox.classList.remove(
            'hidden',
            'is-error',
            'is-success'
        );
    };

    if (!city || !state || !country) {
        showMessage(
            'Please fill all fields.',
            'is-error'
        );
        return;
    }

    loading.classList.remove('hidden');
    hideMessage();
    results.classList.add('hidden');
    results.classList.remove('is-visible');
    alertBox.classList.add('hidden');
    alertBox.innerHTML = '';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                city,
                state,
                country
            })
        });

        const data = await response.json();
        loading.classList.add('hidden');

        if (!data.success) {
            showMessage(
                data.message || 'Location not found.',
                'is-error'
            );
            return;
        }

        hideMessage();

        document.getElementById('location').innerText =
            `${data.location.city},
             ${data.location.state},
             ${data.location.country}`;

        document.getElementById('temperature').innerText =
            `${data.weather.temperature} °C`;

        document.getElementById('humidity').innerText =
            `${data.weather.humidity} %`;

        document.getElementById('rainfall').innerText =
            `${data.weather.rainfall} mm`;

        document.getElementById('wind').innerText =
            `${data.weather.wind_speed} km/h`;

        document.getElementById('flood-risk').innerText =
            data.risks.flood_risk;

        document.getElementById('heat-risk').innerText =
            data.risks.heat_risk;

        let alertsHTML = "";
        data.alerts.forEach(alertMessage => {
            alertsHTML += `
                <div class="notification">
                    ${alertMessage}
                </div>
            `;
        });

        alertBox.innerHTML = alertsHTML;
        alertBox.classList.remove('hidden');

        resultStatus.innerText =
            "Climate analysis completed";
        resultSummary.innerText =
            "Live weather and risk analysis generated successfully.";
        statusPill.innerText =
            "Analysis Complete";

        results.classList.remove('hidden');
        requestAnimationFrame(() => {
            results.classList.add('is-visible');
        });

        // ==========================================
        // 🗺️ NEW CODE: HANDLE THE INTERACTIVE MAP
        // ==========================================
        
        const lat = data.location.latitude || data.location.lat || 0;
        const lon = data.location.longitude || data.location.lon || 0;

        if (lat !== 0 || lon !== 0) {
            setTimeout(() => {
                
                if (!climateMap) {
                    climateMap = L.map('climate-map').setView([lat, lon], 10);
                    
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '&copy; OpenStreetMap contributors'
                    }).addTo(climateMap);
                } else {
                    climateMap.flyTo([lat, lon], 10);
                    
                    climateMap.eachLayer((layer) => {
                        if (!!layer.toGeoJSON && !layer.getTileUrl) {
                            climateMap.removeLayer(layer);
                        }
                    });
                }

                const maxRisk = Math.max(data.risks.flood_risk, data.risks.heat_risk);
                let zoneColor = '#4CAF50'; 

                if (maxRisk >= 0.6) {
                    zoneColor = '#FF3B30'; 
                } else if (maxRisk >= 0.35) {
                    zoneColor = '#FF9500'; 
                }

                L.circle([lat, lon], {
                    color: zoneColor,
                    fillColor: zoneColor,
                    fillOpacity: 0.35,
                    radius: 12000 
                }).addTo(climateMap);

                L.marker([lat, lon])
                    .addTo(climateMap)
                    .bindPopup(`<b>${data.location.city} Threat Assessment Zone</b><br>Flood Risk: ${data.risks.flood_risk}<br>Heat Risk: ${data.risks.heat_risk}`)
                    .openPopup();

                climateMap.invalidateSize();

            }, 300);
        }

    } catch (error) {
        console.error(error);
        loading.classList.add('hidden');
        showMessage(
            'Backend server is not running.',
            'is-error'
        );
    }
}
