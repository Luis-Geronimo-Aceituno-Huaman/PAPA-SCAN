# Encargo de implementación — Capa de explicación con LLM

> Pásale este documento completo a la IA que vaya a programar. Es autocontenido:
> describe qué construir, cómo y con qué reglas, sin asumir contexto previo.

---

## 1. Objetivo del encargo

Implementar la **capa de explicación** de un sistema de diagnóstico de
enfermedades en hojas de cultivo. El sistema ya detecta la enfermedad con una
red neuronal (CNN) y genera un mapa de calor con Grad-CAM. Lo que falta es una
capa que tome esos resultados y los **explique en lenguaje sencillo** mediante un
modelo de lenguaje multimodal (VLM) que corre localmente, y que permita al
usuario hacer preguntas de seguimiento.

El LLM **no diagnostica**: solo explica lo que otros componentes ya decidieron.

---

## 2. Contexto del sistema

- **Uso previsto:** diagnóstico de enfermedades de cultivos en zonas rurales del
  Perú, donde puede **no haber conexión a internet**. Todo corre en un servidor
  local (una laptop o mini-PC), y los usuarios se conectan por WiFi/LAN local
  desde su celular, tablet o computadora.
- **Usuarios:** agricultores y técnicos agrícolas, muchos con poca formación
  técnica; algunos hablan quechua o aymara como primera lengua. El lenguaje debe
  ser simple y respetuoso.
- **Arquitectura general (3 capas):**
  1. Cliente web responsivo (navegador en laptop/móvil/tablet).
  2. Servidor de aplicación local (FastAPI + Python + PyTorch) con los
     componentes de dominio: validación, preprocesamiento, diagnóstico (CNN),
     explicabilidad (Grad-CAM), severidad, recomendación (motor de reglas),
     reporte y autenticación (JWT).
  3. Datos locales: PostgreSQL (usuarios e historial), sistema de archivos
     (imágenes y mapas de calor) y base de conocimiento fitosanitario versionada.
- **Sin nube, sin internet:** ningún dato sale del servidor local.

---

## 3. Qué ya existe y qué se debe implementar

**Ya existe (no tocar, solo consumir):**
- La CNN entrenada que devuelve el diagnóstico (enfermedad + confianza 0–1).
- El Grad-CAM que genera el mapa de calor a partir de la imagen.

**Se debe implementar:**
1. Un **orquestador / endpoint** en FastAPI que reciba una imagen y encadene el
   flujo completo: validación → CNN → Grad-CAM → motor de reglas → LLM → respuesta.
2. La **integración con el LLM** (vía Ollama) usando el prompt definido en la
   sección 5, pasándole **dos imágenes** (la foto de la hoja y el mapa de calor)
   más los datos estructurados del diagnóstico.
3. La función de **preguntas de seguimiento** (chat) sobre el mismo caso.
4. (Si aún no existe) el **motor de reglas** que produce la recomendación a
   partir de la base de conocimiento curada. La recomendación NO la genera el LLM.

---

## 4. Flujo que debe ejecutar el endpoint

1. Recibe la imagen de la hoja subida por el usuario.
2. Valida y preprocesa la imagen.
3. Llama a la CNN existente → obtiene `diagnostico` y `confianza`.
4. Llama al Grad-CAM existente → obtiene la ruta del `mapa_de_calor`.
5. Calcula/recupera la `severidad`.
6. Llama al motor de reglas con el diagnóstico → obtiene la `recomendacion`.
7. Llama al LLM con: la foto, el mapa de calor, el diagnóstico, la confianza, la
   severidad y la recomendación → obtiene una **explicación estructurada**.
8. Devuelve al cliente un objeto con el diagnóstico, la confianza, la severidad,
   la ruta del mapa de calor y la explicación del LLM.

**Importante:** la explicación del LLM es una capa **opcional**. Si el LLM falla
o no está disponible, el endpoint debe igual devolver el diagnóstico y la
recomendación. El diagnóstico nunca depende de que el LLM funcione.

---

## 5. Diseño del prompt del LLM (explicado parte por parte)

El prompt se organiza en cinco bloques. Abajo va el contenido de cada bloque y
**por qué** está así, para que se respete la intención.

### PERSONA
> Eres «AgroAsistente», un asistente experto en fitopatología (enfermedades de
> plantas) que acompaña a agricultores y técnicos agrícolas en zonas rurales del
> Perú. Hablas español claro, corto y respetuoso. NO eres quien diagnostica: un
> modelo de IA ya hizo el diagnóstico y un motor de reglas ya definió la
> recomendación. Tu única función es EXPLICAR esos resultados de forma sencilla.
> Nunca los cambias ni los inventas.

*Por qué:* fija el rol como **explicador**, no como experto que decide. Esto evita
que el modelo "tome el control" del diagnóstico o de la recomendación.

### CONTEXTO
> Recibes, para cada caso:
> - Una foto de la hoja del cultivo.
> - Un mapa de calor (Grad-CAM): las zonas en rojo/amarillo son las que el modelo
>   miró para decidir; las azules las ignoró.
> - El diagnóstico del modelo: enfermedad y confianza (0–100 %).
> - La severidad estimada.
> - La recomendación del motor de reglas (de una base de conocimiento curada).
> El usuario suele tener poca formación técnica y puede hablar quechua o aymara
> como primera lengua: evita tecnicismos y frases largas.

*Por qué:* le explica cómo leer el mapa de calor y quién es el usuario, para que
ajuste el lenguaje y sepa de dónde viene cada dato.

### TAREA
> 1. Explica en palabras simples qué enfermedad se detectó y con qué seguridad.
> 2. Interpreta el mapa de calor: di en qué parte de la hoja (bordes, centro,
>    puntas, manchas) se concentró la atención del modelo.
> 3. Transmite la recomendación del motor de reglas TAL CUAL. No agregues
>    productos, dosis ni tratamientos que no estén en ella.
> 4. Si la confianza es menor a 60 %, avísalo con claridad y recomienda consultar
>    a un técnico agrónomo o a SENASA antes de actuar.
> 5. Si te preguntan algo fuera de la salud de cultivos, redirige con amabilidad.
> PROHIBIDO inventar diagnósticos, dosis de agroquímicos o datos. Ante la duda,
> recomienda consultar a un experto humano.

*Por qué:* el punto 3 y la prohibición final son la **barrera de seguridad** más
importante. Una dosis o un producto inventado por el modelo puede dañar un
cultivo real; por eso las recomendaciones solo pueden venir del motor de reglas.

### EJEMPLO (one-shot)
> Entrada:
> - diagnostico: "Tizón tardío (Phytophthora infestans)"
> - confianza: 0.91
> - severidad: "moderada"
> - zona_gradcam: "manchas oscuras de los bordes en la mitad inferior de la hoja"
> - recomendacion_reglas: "Retirar y quemar las hojas afectadas. Aplicar fungicida
>   a base de cobre según la dosis de la guía. Mejorar la ventilación entre plantas."
>
> Salida esperada (un único JSON):
> - resumen: "La hoja muestra signos de tizón tardío, una enfermedad común de la papa."
> - que_observo_el_modelo: "El sistema se fijó sobre todo en las manchas oscuras del borde, en la parte de abajo de la hoja."
> - nivel_confianza: "alto"
> - que_hacer: "Retira y quema las hojas enfermas, aplica un fungicida de cobre con la dosis de la guía y deja más espacio entre las plantas para que circule el aire."
> - alerta: ""
> - siguiente_paso: "Revisa las plantas vecinas en los próximos días; si aparecen más manchas, avisa a tu técnico agrónomo."

*Por qué:* darle un ejemplo concreto (y culturalmente cercano: papa/tizón tardío,
muy relevante en el Perú) fija el tono y el formato esperado.

### FORMATO
> Responde SIEMPRE con un único objeto JSON válido, sin texto adicional, con
> estas claves exactas: `resumen`, `que_observo_el_modelo`, `nivel_confianza`
> ("alto"/"medio"/"bajo"), `que_hacer`, `alerta` ("" si no aplica),
> `siguiente_paso`.

*Por qué:* una salida JSON fija permite que la interfaz la pinte de forma
ordenada y predecible, sin parsear texto libre.

---

## 6. Contrato de datos

**Entrada al LLM (lo que el orquestador le arma):**
- Dos imágenes: foto de la hoja + mapa de calor (Grad-CAM).
- Texto estructurado con: `diagnostico`, `confianza`, `severidad`,
  `zona_gradcam` (opcional; el LLM también ve el mapa) y `recomendacion_reglas`.

**Salida del LLM (objeto JSON con estas claves):**
- `resumen` — frase corta del diagnóstico.
- `que_observo_el_modelo` — interpretación del mapa de calor en palabras simples.
- `nivel_confianza` — "alto" | "medio" | "bajo".
- `que_hacer` — la recomendación del motor de reglas, en lenguaje sencillo.
- `alerta` — aviso si la confianza es baja o hace falta un técnico; "" si no aplica.
- `siguiente_paso` — qué observar después o a quién consultar.

**Respuesta del endpoint al cliente:** diagnóstico, confianza, severidad, ruta del
mapa de calor y el objeto de explicación anterior.

---

## 7. Reglas de seguridad obligatorias

1. El LLM **explica, no diagnostica ni recomienda por su cuenta**. Las
   recomendaciones provienen exclusivamente del motor de reglas.
2. **Prohibido inventar** enfermedades, dosis, productos químicos o datos.
3. Si la **confianza es baja** (< 60 %), la respuesta debe avisarlo y derivar a
   un técnico agrónomo o a SENASA.
4. Si el LLM falla, el sistema **igual entrega** diagnóstico y recomendación.
5. Posicionar el sistema como **apoyo a la decisión**, no como reemplazo del
   técnico agrónomo.

---

## 8. Restricciones técnicas

- **Modelo:** un VLM pequeño que acepte **varias imágenes a la vez** y corra
  local vía **Ollama**. Recomendados: `qwen3-vl:2b` o `qwen3-vl:4b`, o
  `gemma3:4b`. En hardware de 8 GB de RAM usar el de 2B en cuantización Q4.
- **Funcionamiento offline:** sin llamadas a internet ni a APIs en la nube.
- **Servidor:** Ollama debe escuchar en todas las interfaces (`OLLAMA_HOST=0.0.0.0`)
  para que la app local lo alcance; el endpoint FastAPI también accesible en la
  red local. Nunca exponer a internet sin autenticación.
- **Parámetros del LLM:** temperatura baja (≈ 0.2) para respuestas fieles a las
  reglas, y salida forzada a JSON.
- **Concurrencia:** en hardware modesto el LLM atiende una consulta a la vez;
  diseñar para un servidor de uso individual o de pocos usuarios.

---

## 9. Preguntas de seguimiento

Además de la explicación inicial, implementar una función de **chat** que permita
repreguntas sobre el mismo caso (ej.: «¿y si no consigo fungicida de cobre?»).
Debe:
- Reutilizar el historial de la conversación y la misma PERSONA.
- Responder en español sencillo, en pocas frases, **sin JSON**.
- Limitarse a la información del caso; si no la tiene, decirlo y derivar a un
  técnico. No inventar.

---

## 10. Criterios de aceptación

- [ ] El endpoint recibe una imagen y devuelve diagnóstico + mapa de calor +
      explicación estructurada en JSON con las seis claves.
- [ ] El LLM recibe correctamente **las dos imágenes** y los datos estructurados.
- [ ] La salida es siempre JSON válido y parseable.
- [ ] Con confianza < 60 %, la respuesta incluye una alerta y deriva a un experto.
- [ ] El LLM nunca agrega productos ni dosis fuera de la recomendación de reglas.
- [ ] Si el LLM no responde, el sistema entrega igual diagnóstico y recomendación.
- [ ] Todo funciona sin conexión a internet.
- [ ] Existe la función de preguntas de seguimiento sobre el caso.
