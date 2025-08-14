# DocuChat - Copiloto Conversacional sobre Documentos

Aplicación desarrollada para el desafío técnico de CatchAI que permite conversar inteligentemente con hasta 5 documentos PDF mediante RAG.

## Instalación y Ejecución

### Requisitos Previos
- Docker y Docker Compose instalados
- API Key de Google Gemini ([Obtener aquí](https://aistudio.google.com/app/apikey))

### Ejecución con Docker
```bash
docker-compose up --build
```

La aplicación estará disponible en: http://localhost:8501

## Arquitectura del Sistema

![Arquitectura](/arquitectura.svg)

### Componentes Principales:

1. **Frontend**: Streamlit para interfaz conversacional
2. **Orquestación**: LangChain para manejo de cadenas RAG
3. **Vector Store**: ChromaDB para almacenamiento de embeddings
4. **Embeddings**: Google Gemini Embeddings
5. **LLM**: Google Gemini 2.5 Flash

## ⚙️ Justificación de Elecciones Técnicas

### LangChain
- **Ventaja**: Ecosistema maduro para aplicaciones LLM
- **Funcionalidad**: Chains pre-construidas para RAG
- **Escalabilidad**: Fácil integración con diferentes LLMs

### ChromaDB
- **Simplicidad**: Base de datos vectorial ligera
- **Performance**: Búsqueda rápida de similitudes
- **Memoria**: Gestión automática de colecciones

### Google Gemini
- **Costo-efectivo**: Tier gratuito generoso
- **Multimodal**: Capacidad de procesar texto y PDFs
- **Calidad**: Respuestas coherentes y contextuales

### Streamlit
- **Rapidez de desarrollo**: Prototipado rápido
- **Interactividad**: Componentes reactivos nativos
- **Deployment**: Fácil despliegue y compartir

## 🔄 Flujo Conversacional

1. **Carga de Documentos**:
   ```
   PDF Upload → PyPDFLoader → Text Splitting → Embeddings → ChromaDB
   ```

2. **Procesamiento de Query**:
   ```
   User Question → History-Aware Retriever → Context Retrieval → LLM Response
   ```

3. **RAG Pipeline**:
   ```
   Query + History → Retrieval Chain → Document Chunks → Prompt Template → LLM → Response
   ```

## ✨ Funcionalidades Implementadas

### Requisitos Mínimos ✅
- [x] Subida de hasta 5 PDFs
- [x] Extracción y vectorización de contenido  
- [x] Interfaz conversacional
- [x] Orquestación estructurada con LangChain

### Funcionalidades Opcionales ✅
- [x] **Resumen automático** de documentos cargados
- [x] **Comparación inteligente** entre múltiples documentos
- [x] **Clasificación por temas** y análisis temático
- [x] **Historial conversacional** con context-awareness
- [x] **Gestión de memoria** para optimización de rendimiento

### Parámetros Configurables
- **Chunk Size**: 5000 caracteres (optimizado para PDFs)
- **Chunk Overlap**: 1000 caracteres (preserva contexto)
- **Temperature**: 0.3 (balance creatividad/precisión)
- **Max Collections**: 20 (gestión de memoria)

## 🚧 Limitaciones Actuales

1. **Limite de Procesamiento**: La capa gratuita limita la velocidad de subida de archivos
2. **Límite de archivos**: Máximo 5 PDFs simultáneos
3. **Persistencia**: Base de datos en memoria (se resetea al reiniciar)
4. **Idioma**: Optimizado para español
5. **Concurrencia**: Una sesión por instancia

## 🗺️ Roadmap y Mejoras Futuras

### Corto Plazo (1-2 semanas)
- [ ] Persistencia de base de datos vectorial
- [ ] Gemini API Key Tier 1 
- [ ] Métricas de calidad de respuestas
- [ ] Caché de embeddings para optimización

### Mediano Plazo (1-2 meses)  
- [ ] Autenticación de usuarios
- [ ] Múltiples sesiones concurrentes
- [ ] API REST para integración externa
- [ ] Dashboard de analytics

### Largo Plazo (3+ meses)
- [ ] Multi-tenancy empresarial
- [ ] Integración con bases de datos empresariales
- [ ] Soporte para documentos multimedia
- [ ] Fine-tuning de modelos específicos

---

**Desarrollado por**: [Alex Tardon](https://github.com/AlexFT257)
**Fecha**: 13 Agosto 2025  
**Desafío**: CatchAI - Desarrollador/a Joven de IA