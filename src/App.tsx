import * as toxicity from "@tensorflow-models/toxicity";
import { useState, useEffect, useRef } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";

interface ToxicityResult {
  label: string;
  probability: number;
  match: boolean | null;
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
}

interface ToxicityResult {
  label: string;
  probability: number;
  match: boolean | null;
}

type ViewType = "toxicity" | "trucks";

function App() {
  const [currentView, setCurrentView] = useState<ViewType>("toxicity");
  const [model, setModel] = useState<toxicity.ToxicityClassifier | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);

  const [inputText, setInputText] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<
    ToxicityResult[] | null
  >(null);

  // Carga inicial del modelo
  useEffect(() => {
    const loadModel = async () => {
      try {
        const threshold = 0.6;
        const loadedModel = await toxicity.load(threshold, []);
        setModel(loadedModel);
      } catch (error) {
        console.error("Error catastrófico cargando el modelo:", error);
      } finally {
        setIsModelLoading(false);
      }
    };

    loadModel();
  }, []);

  const [ssdModel, setSsdModel] = useState<cocoSsd.ObjectDetection | null>(
    null,
  );
  const [isSsdLoading, setIsSsdLoading] = useState(true);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [truckBoxes, setTruckBoxes] = useState<BoundingBox[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);

  const imageRef = useRef<HTMLImageElement>(null);

  // Carga del modelo COCO-SSD
  useEffect(() => {
    const loadSsdModel = async () => {
      try {
        const loadedModel = await cocoSsd.load();
        setSsdModel(loadedModel);
      } catch (error) {
        console.error("Error cargando COCO-SSD:", error);
      } finally {
        setIsSsdLoading(false);
      }
    };
    loadSsdModel();
  }, []);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setTruckBoxes([]);
      const imageUrl = URL.createObjectURL(file);
      setImageSrc(imageUrl);
    }
  };

  const detectTrucks = async () => {
    if (!ssdModel || !imageRef.current) return;

    setIsDetecting(true);
    setTruckBoxes([]);

    try {
      const predictions = await ssdModel.detect(imageRef.current);

      const foundTrucks = predictions
        .filter((p) => p.class === "truck")
        .map((p) => ({
          x: p.bbox[0],
          y: p.bbox[1],
          width: p.bbox[2],
          height: p.bbox[3],
          class: p.class,
          score: Math.round(p.score * 100),
        }));

      setTruckBoxes(foundTrucks);
    } catch (error) {
      console.error("Error en la detección visual:", error);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleAnalyze = async () => {
    if (!model || !inputText.trim()) return;

    setIsEvaluating(true);
    setAnalysisResults(null);

    try {
      const predictions = await model.classify([inputText]);

      const detailedResults: ToxicityResult[] = predictions.map((p) => {
        const probValue = p.results[0].probabilities[1] as number;

        return {
          label: p.label,
          probability: Math.round(probValue * 100),
          match: p.results[0].match,
        };
      });

      detailedResults.sort((a, b) => b.probability - a.probability);

      setAnalysisResults(detailedResults);
    } catch (error) {
      console.error("Error durante la inferencia:", error);
    } finally {
      setIsEvaluating(false);
    }
  };
  return (
    <div
      style={{
        maxWidth: "800px",
        margin: "0 auto",
        padding: "20px",
        fontFamily: "system-ui",
      }}
    >
      <h1>TP Seminario: TensorFlow.js</h1>

      <nav
        style={{
          display: "flex",
          gap: "10px",
          marginBottom: "20px",
          paddingBottom: "20px",
          borderBottom: "2px solid #eee",
        }}
      >
        <button
          onClick={() => setCurrentView("toxicity")}
          style={{
            padding: "10px 20px",
            cursor: "pointer",
            backgroundColor: currentView === "toxicity" ? "#007bff" : "#f8f9fa",
            color: currentView === "toxicity" ? "white" : "black",
            border: "1px solid #ccc",
            borderRadius: "5px",
            fontWeight: "bold",
          }}
        >
          1. Análisis de Texto (Toxicidad)
        </button>

        <button
          onClick={() => setCurrentView("trucks")}
          style={{
            padding: "10px 20px",
            cursor: "pointer",
            backgroundColor: currentView === "trucks" ? "#28a745" : "#f8f9fa",
            color: currentView === "trucks" ? "white" : "black",
            border: "1px solid #ccc",
            borderRadius: "5px",
            fontWeight: "bold",
          }}
        >
          2. Visión Computacional (Camiones)
        </button>
      </nav>

      <main>
        {currentView === "toxicity" && (
          <div>
            <h2>Sección de Toxicidad Activa</h2>

            <div
              style={{
                maxWidth: "600px",
                margin: "0 auto",
                padding: "20px",
                fontFamily: "system-ui",
              }}
            >
              <hr />

              <section style={{ marginBottom: "40px" }}>
                <h2>1. Detección de Toxicidad</h2>

                {isModelLoading ? (
                  <div
                    style={{
                      padding: "20px",
                      backgroundColor: "#fff3cd",
                      color: "#856404",
                      borderRadius: "5px",
                    }}
                  >
                    <p>
                      Descargando y cargando modelo neuronal en memoria (pesa
                      unos megas, paciencia)...
                    </p>
                  </div>
                ) : (
                  <div>
                    <textarea
                      style={{
                        width: "100%",
                        padding: "10px",
                        fontSize: "16px",
                        borderRadius: "5px",
                      }}
                      rows={4}
                      placeholder="Escribí una frase acá para analizar..."
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      disabled={isEvaluating}
                    />
                    <button
                      onClick={handleAnalyze}
                      disabled={isEvaluating || !inputText.trim()}
                      style={{
                        marginTop: "10px",
                        padding: "10px 20px",
                        fontSize: "16px",
                        cursor: "pointer",
                      }}
                    >
                      {isEvaluating
                        ? "Analizando tensores..."
                        : "Analizar Texto"}
                    </button>

                    {analysisResults !== null && (
                      <div
                        style={{
                          marginTop: "20px",
                          padding: "15px",
                          borderRadius: "5px",
                          border: "1px solid #ccc",
                        }}
                      >
                        <h3 style={{ marginTop: 0 }}>
                          Estadísticas del Modelo:
                        </h3>
                        <table
                          style={{
                            width: "100%",
                            textAlign: "left",
                            borderCollapse: "collapse",
                          }}
                        >
                          <thead>
                            <tr style={{ borderBottom: "2px solid #ddd" }}>
                              <th style={{ padding: "8px" }}>
                                Categoría (Label)
                              </th>
                              <th style={{ padding: "8px" }}>Probabilidad</th>
                              <th style={{ padding: "8px" }}>
                                Veredicto ({">"}60%)
                              </th>{" "}
                            </tr>
                          </thead>
                          <tbody>
                            {analysisResults.map((item) => (
                              <tr
                                key={item.label}
                                style={{ borderBottom: "1px solid #eee" }}
                              >
                                <td
                                  style={{
                                    padding: "8px",
                                    textTransform: "capitalize",
                                  }}
                                >
                                  {item.label.replace("_", " ")}
                                </td>
                                <td style={{ padding: "8px" }}>
                                  <div
                                    style={{
                                      display: "flex",
                                      alignItems: "center",
                                      gap: "10px",
                                    }}
                                  >
                                    <span style={{ width: "40px" }}>
                                      {item.probability}%
                                    </span>
                                    <div
                                      style={{
                                        flex: 1,
                                        backgroundColor: "#e9ecef",
                                        borderRadius: "4px",
                                        height: "10px",
                                      }}
                                    >
                                      <div
                                        style={{
                                          width: `${item.probability}%`,
                                          backgroundColor: item.match
                                            ? "#dc3545"
                                            : "#007bff",
                                          height: "100%",
                                          borderRadius: "4px",
                                        }}
                                      ></div>
                                    </div>
                                  </div>
                                </td>
                                <td
                                  style={{
                                    padding: "8px",
                                    fontWeight: "bold",
                                    color: item.match ? "#dc3545" : "#28a745",
                                  }}
                                >
                                  {item.match ? "DETECTADO" : "Limpio"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </section>
            </div>
          </div>
        )}

        {currentView === "trucks" && (
          <div>
            <h2>Sección de Camiones Activa</h2>
            {currentView === "trucks" && (
              <section>
                <h2>2. Detección de Camiones</h2>

                {isSsdLoading ? (
                  <div
                    style={{
                      padding: "20px",
                      backgroundColor: "#e2e3e5",
                      borderRadius: "5px",
                    }}
                  >
                    <p>Cargando modelo de Visión Computacional (COCO-SSD)...</p>
                  </div>
                ) : (
                  <div>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      style={{ marginBottom: "15px", display: "block" }}
                    />

                    {imageSrc && (
                      <div style={{ marginBottom: "15px" }}>
                        <button
                          onClick={detectTrucks}
                          disabled={isDetecting}
                          style={{
                            padding: "10px 20px",
                            fontSize: "16px",
                            cursor: "pointer",
                            backgroundColor: "#28a745",
                            color: "white",
                            border: "none",
                            borderRadius: "5px",
                          }}
                        >
                          {isDetecting
                            ? "Escaneando píxeles..."
                            : "Detectar Camión"}
                        </button>
                      </div>
                    )}

                    {/* Contenedor relativo para que las cajas se dibujen arriba de la foto */}
                    <div
                      style={{ position: "relative", display: "inline-block" }}
                    >
                      {imageSrc && (
                        <img
                          ref={imageRef}
                          src={imageSrc}
                          alt="Subida para analizar"
                          style={{
                            maxWidth: "100%",
                            height: "auto",
                            borderRadius: "5px",
                          }}
                        />
                      )}

                      {truckBoxes.map((box, index) => (
                        <div
                          key={index}
                          style={{
                            position: "absolute",
                            left: `${box.x}px`,
                            top: `${box.y}px`,
                            width: `${box.width}px`,
                            height: `${box.height}px`,
                            border: "3px solid #dc3545",
                            backgroundColor: "rgba(220, 53, 69, 0.2)",
                            pointerEvents: "none",
                          }}
                        >
                          <span
                            style={{
                              position: "absolute",
                              top: "-25px",
                              left: "-3px",
                              backgroundColor: "#dc3545",
                              color: "white",
                              padding: "2px 5px",
                              fontSize: "14px",
                              fontWeight: "bold",
                            }}
                          >
                            Camión ({box.score}%)
                          </span>
                        </div>
                      ))}
                    </div>

                    {imageSrc && truckBoxes.length === 0 && !isDetecting && (
                      <p style={{ color: "#856404", marginTop: "10px" }}>
                        No se detectaron camiones en esta imagen.
                      </p>
                    )}
                  </div>
                )}
              </section>
            )}{" "}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
