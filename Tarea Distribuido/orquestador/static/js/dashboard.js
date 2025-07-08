/**
 * Dashboard Koch - JavaScript para funcionalidad interactiva
 */

let autoRefreshEnabled = true;
let refreshInterval;
let estadosChart, cargaChart;

// Inicializar gráficos
function inicializarGraficos() {
    // Gráfico de estado de trabajos
    const ctxEstados = document.getElementById('estadosChart').getContext('2d');
    estadosChart = new Chart(ctxEstados, {
        type: 'bar',
        data: {
            labels: ['Completados', 'Fallidos', 'En Cola', 'En Proceso'],
            datasets: [{
                label: 'Cantidad de Trabajos',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    '#28a745',  // Verde para completados
                    '#dc3545',  // Rojo para fallidos
                    '#ffc107',  // Amarillo para en cola
                    '#007bff'   // Azul para en proceso
                ],
                borderColor: [
                    '#1e7e34',
                    '#c82333',
                    '#e0a800',
                    '#0056b3'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Estado Actual del Sistema'
                }
            }
        }
    });
    
    // Gráfico de carga por esclavo
    const ctxCarga = document.getElementById('cargaChart').getContext('2d');
    cargaChart = new Chart(ctxCarga, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Trabajos Ejecutados',
                    data: [],
                    backgroundColor: '#007bff',
                    borderColor: '#0056b3',
                    borderWidth: 1
                },
                {
                    label: 'Tiempo Total (s)',
                    data: [],
                    backgroundColor: '#28a745',
                    borderColor: '#1e7e34',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    },
                    title: {
                        display: true,
                        text: 'Trabajos Ejecutados'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Tiempo Total (s)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Carga de Trabajo por Esclavo'
                }
            }
        }
    });
}

// Actualizar gráficos con nuevos datos
function actualizarGraficos(data) {
    // Actualizar gráfico de estados
    const stats = data.estadisticas;
    estadosChart.data.datasets[0].data = [
        stats.trabajos_completados,
        stats.trabajos_fallidos,
        data.cola_trabajos,
        data.trabajos_en_proceso
    ];
    estadosChart.update('none'); // Sin animación para mejor performance
    
    // Actualizar gráfico de carga por esclavo
    const esclavos = data.esclavos;
    const esclavosIds = Object.keys(esclavos);
    const trabajosEjecutados = esclavosIds.map(id => esclavos[id].trabajos_ejecutados);
    const tiempoTotal = esclavosIds.map(id => Math.round(esclavos[id].tiempo_total));
    
    // Actualizar labels y datos
    cargaChart.data.labels = esclavosIds.map(id => {
        const status = esclavos[id].status;
        const emoji = status === 'activo' ? '✅' : status === 'inactivo' ? '❌' : '⚠️';
        return `${emoji} ${id}`;
    });
    
    cargaChart.data.datasets[0].data = trabajosEjecutados;
    cargaChart.data.datasets[1].data = tiempoTotal;
    
    // Cambiar color de barras según estado del esclavo
    cargaChart.data.datasets[0].backgroundColor = esclavosIds.map(id => {
        const status = esclavos[id].status;
        return status === 'activo' ? '#007bff' : 
               status === 'inactivo' ? '#6c757d' : '#ffc107';
    });
    
    cargaChart.data.datasets[1].backgroundColor = esclavosIds.map(id => {
        const status = esclavos[id].status;
        return status === 'activo' ? '#28a745' : 
               status === 'inactivo' ? '#6c757d' : '#fd7e14';
    });
    
    cargaChart.update('none');
}

// Función para actualizar datos sin recargar la página
async function actualizarDatos() {
    if (!autoRefreshEnabled) return;
    
    try {
        // Actualizar estadísticas generales
        const response = await fetch('/status');
        const data = await response.json();
        
        // Actualizar estadísticas
        document.getElementById('trabajos-completados').textContent = data.estadisticas.trabajos_completados;
        document.getElementById('trabajos-fallidos').textContent = data.estadisticas.trabajos_fallidos;
        document.getElementById('tiempo-total').textContent = data.estadisticas.tiempo_total_procesamiento.toFixed(2) + 's';
        document.getElementById('cola-size').textContent = data.cola_trabajos;
        document.getElementById('trabajos-proceso').textContent = data.trabajos_en_proceso;
        
        // Actualizar tabla de esclavos
        actualizarTablaEsclavos(data.esclavos);
        
        // Actualizar trabajos en proceso
        await actualizarTrabajosEnProceso();
        
        // Actualizar últimos resultados
        await actualizarUltimosResultados();
        
        // Actualizar gráficos
        actualizarGraficos(data);
        
        // Actualizar indicador de refresh
        document.getElementById('refreshStatus').textContent = '🔄 Actualizado: ' + new Date().toLocaleTimeString();
        
    } catch (error) {
        console.error('Error actualizando datos:', error);
        document.getElementById('refreshStatus').textContent = '❌ Error de conexión';
    }
}

function actualizarTablaEsclavos(esclavos) {
    let html = `
        <table>
            <tr>
                <th>ID</th>
                <th>URL</th>
                <th>Estado</th>
                <th>Último Ping</th>
                <th>Trabajos Ejecutados</th>
                <th>Tiempo Total</th>
            </tr>`;
    
    for (const [esclavo_id, info] of Object.entries(esclavos)) {
        const lastPing = info.last_ping ? new Date(info.last_ping).toLocaleTimeString() : 'Nunca';
        html += `
            <tr>
                <td>${esclavo_id}</td>
                <td>${info.url}</td>
                <td class="status-${info.status}">${info.status}</td>
                <td>${lastPing}</td>
                <td>${info.trabajos_ejecutados}</td>
                <td>${info.tiempo_total.toFixed(2)}s</td>
            </tr>`;
    }
    
    html += '</table>';
    document.getElementById('esclavos-table').innerHTML = html;
}

async function actualizarTrabajosEnProceso() {
    try {
        const response = await fetch('/trabajos_en_proceso');
        const trabajos = await response.json();
        
        let html = '';
        
        if (Object.keys(trabajos).length > 0) {
            html = `
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Esclavo</th>
                        <th>Algoritmo</th>
                        <th>Parámetros</th>
                        <th>Tiempo Ejecutando</th>
                    </tr>`;
            
            for (const [trabajo_id, info] of Object.entries(trabajos)) {
                const parametros = info.trabajo.parametros;
                html += `
                    <tr>
                        <td>${trabajo_id.substring(0, 8)}...</td>
                        <td>${info.esclavo_id}</td>
                        <td>${info.trabajo.algoritmo}</td>
                        <td>iter=${parametros.iteraciones}, size=${parametros.size}</td>
                        <td>${info.tiempo_ejecutando.toFixed(1)}s</td>
                    </tr>`;
            }
            
            html += '</table>';
        } else {
            html = '<p>No hay trabajos en proceso actualmente.</p>';
        }
        
        document.getElementById('trabajos-proceso-content').innerHTML = html;
    } catch (error) {
        console.error('Error actualizando trabajos en proceso:', error);
    }
}

async function actualizarUltimosResultados() {
    try {
        const response = await fetch('/ultimos_resultados');
        const resultados = await response.json();
        
        let html = '';
        
        if (Object.keys(resultados).length > 0) {
            html = `
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Esclavo</th>
                        <th>Estado</th>
                        <th>Tiempo</th>
                        <th>Timestamp</th>
                    </tr>`;
            
            for (const [trabajo_id, resultado] of Object.entries(resultados)) {
                const estado = resultado.error ? '❌ Error' : '✅ Exitoso';
                const tiempo = resultado.tiempo_ejecucion ? resultado.tiempo_ejecucion.toFixed(2) + 's' : 'N/A';
                const esclavo = resultado.esclavo_id || 'N/A';
                
                html += `
                    <tr>
                        <td>${trabajo_id.substring(0, 8)}...</td>
                        <td>${esclavo}</td>
                        <td>${estado}</td>
                        <td>${tiempo}</td>
                        <td>${new Date(resultado.timestamp).toLocaleString()}</td>
                    </tr>`;
            }
            
            html += '</table>';
        } else {
            html = '<p>No hay resultados todavía.</p>';
        }
        
        document.getElementById('resultados-content').innerHTML = html;
    } catch (error) {
        console.error('Error actualizando resultados:', error);
    }
}

function toggleAutoRefresh() {
    autoRefreshEnabled = !autoRefreshEnabled;
    const button = document.getElementById('refreshStatus');
    
    if (autoRefreshEnabled) {
        button.textContent = '🔄 Auto-actualización: ON';
        button.style.background = '#007bff';
        iniciarAutoRefresh();
    } else {
        button.textContent = '⏸️ Auto-actualización: OFF';
        button.style.background = '#6c757d';
        clearInterval(refreshInterval);
    }
}

function iniciarAutoRefresh() {
    // Limpiar intervalo existente
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    
    // Iniciar nuevo intervalo
    refreshInterval = setInterval(actualizarDatos, 3000); // Cada 3 segundos
}

function limpiarFormulario() {
    document.getElementById('algoritmo').value = 'ClassicKoch';
    document.getElementById('iteraciones').value = '4';
    document.getElementById('size').value = '3.0';
    document.getElementById('dpi').value = '300';
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Click en botón de auto-refresh
    document.getElementById('refreshStatus').addEventListener('click', toggleAutoRefresh);
    
    // Manejar envío de formulario con AJAX para evitar recarga
    document.getElementById('form-trabajo').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        try {
            const response = await fetch('/trabajo', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                // Mostrar mensaje de éxito
                alert('✅ Trabajo agregado exitosamente');
                // Actualizar datos inmediatamente
                actualizarDatos();
            } else {
                alert('❌ Error agregando trabajo');
            }
        } catch (error) {
            alert('❌ Error de conexión: ' + error.message);
        }
    });
});

// Iniciar auto-refresh al cargar la página
window.addEventListener('load', function() {
    inicializarGraficos();
    iniciarAutoRefresh();
    // Actualización inicial después de 1 segundo
    setTimeout(actualizarDatos, 1000);
});
