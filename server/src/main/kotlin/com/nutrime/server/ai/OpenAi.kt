package com.nutrime.server.ai

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.future.await
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.ArrayDeque

/**
 * OpenAI Responses API adapter. The rest of the agent keeps its provider-neutral model loop in the
 * existing Gemini-shaped internal format; this class translates messages, images and function calls.
 */
class OpenAiClient(
    private val apiKey: String,
    private val model: String,
    private val baseUrl: String = "https://api.openai.com/v1",
    private val reasoningEffort: String? = null,
) : LlmClient {
    private val http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build()
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override fun stream(request: JsonObject): Flow<JsonObject> = flow {
        val body = openAiRequest(request).toString()
        var attempt = 0
        while (true) {
            attempt++
            val req = HttpRequest.newBuilder(URI("${baseUrl.trimEnd('/')}/responses"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer $apiKey")
                .timeout(Duration.ofSeconds(180))
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build()
            val res = http.send(req, HttpResponse.BodyHandlers.ofString())
            if (res.statusCode() !in 200..299) {
                val message = errorMessage(res.body())
                val retryable = res.statusCode() == 429 || res.statusCode() >= 500
                if (retryable && attempt < 2) {
                    delay(1500)
                    continue
                }
                throw LlmException("OpenAI error ${res.statusCode()}: $message", retryable)
            }
            emit(asGeminiChunk(json.parseToJsonElement(res.body()).jsonObject))
            return@flow
        }
    }.flowOn(Dispatchers.IO)

    internal fun openAiRequest(request: JsonObject): JsonObject {
        val instructions = ((request["systemInstruction"] as? JsonObject)?.get("parts") as? JsonArray).orEmpty()
            .mapNotNull { ((it as? JsonObject)?.get("text") as? JsonPrimitive)?.content }
            .joinToString("\n")
        val tools = ((request["tools"] as? JsonArray)?.firstOrNull() as? JsonObject)
            ?.get("functionDeclarations") as? JsonArray ?: JsonArray(emptyList())
        return buildJsonObject {
            put("model", model)
            put("instructions", instructions)
            put("input", responseInput((request["contents"] as? JsonArray).orEmpty()))
            putJsonArray("tools") {
                tools.mapNotNull { it as? JsonObject }.forEach { declaration ->
                    addJsonObject {
                        put("type", "function")
                        put("name", declaration["name"] ?: JsonPrimitive(""))
                        declaration["description"]?.let { put("description", it) }
                        put("parameters", normalizeSchema(declaration["parameters"] ?: buildJsonObject {
                            put("type", "object"); putJsonObject("properties") {}
                        }))
                        put("strict", false)
                    }
                }
            }
            put("tool_choice", "auto")
            put("parallel_tool_calls", true)
            put("store", false)
            put("max_output_tokens", 4096)
            reasoningEffort?.let { effort -> putJsonObject("reasoning") { put("effort", effort) } }
        }
    }

    private fun responseInput(contents: List<JsonElement>): JsonArray {
        val input = mutableListOf<JsonElement>()
        val pendingCalls = mutableMapOf<String, ArrayDeque<String>>()
        var sequence = 0
        contents.mapNotNull { it as? JsonObject }.forEach { content ->
            val role = (content["role"] as? JsonPrimitive)?.content.orEmpty()
            val messageParts = mutableListOf<JsonObject>()
            fun flushMessage() {
                if (messageParts.isEmpty()) return
                input += buildJsonObject {
                    put("type", "message")
                    put("role", if (role == "model") "assistant" else "user")
                    put("status", "completed")
                    putJsonArray("content") { messageParts.forEach { add(it) } }
                }
                messageParts.clear()
            }
            (content["parts"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }.forEach { part ->
                when {
                    part["text"] is JsonPrimitive -> messageParts += buildJsonObject {
                        put("type", if (role == "model") "output_text" else "input_text")
                        put("text", part["text"]!!)
                        if (role == "model") putJsonArray("annotations") {}
                    }
                    part["inlineData"] is JsonObject -> {
                        val image = part["inlineData"]!!.jsonObject
                        val mime = image["mimeType"]?.jsonPrimitive?.content ?: "image/jpeg"
                        val data = image["data"]?.jsonPrimitive?.content.orEmpty()
                        messageParts += buildJsonObject {
                            put("type", "input_image")
                            put("image_url", "data:$mime;base64,$data")
                            put("detail", "auto")
                        }
                    }
                    part["functionCall"] is JsonObject -> {
                        flushMessage()
                        val call = part["functionCall"]!!.jsonObject
                        val name = call["name"]?.jsonPrimitive?.content.orEmpty()
                        val callId = call["id"]?.jsonPrimitive?.content ?: "call_nutrime_${sequence++}"
                        pendingCalls.getOrPut(name) { ArrayDeque() }.addLast(callId)
                        input += buildJsonObject {
                            put("type", "function_call")
                            put("call_id", callId)
                            put("name", name)
                            put("arguments", (call["args"] as? JsonObject ?: JsonObject(emptyMap())).toString())
                        }
                    }
                    part["functionResponse"] is JsonObject -> {
                        flushMessage()
                        val response = part["functionResponse"]!!.jsonObject
                        val name = response["name"]?.jsonPrimitive?.content.orEmpty()
                        val callId = response["id"]?.jsonPrimitive?.content
                            ?: pendingCalls[name]?.pollFirst()
                            ?: "call_nutrime_${sequence++}"
                        input += buildJsonObject {
                            put("type", "function_call_output")
                            put("call_id", callId)
                            put("output", (response["response"] ?: JsonObject(emptyMap())).toString())
                        }
                    }
                }
            }
            flushMessage()
        }
        return JsonArray(input)
    }

    internal fun asGeminiChunk(response: JsonObject): JsonObject {
        val parts = buildJsonArray {
            (response["output"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }.forEach { item ->
                when ((item["type"] as? JsonPrimitive)?.content) {
                    "message" -> (item["content"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }.forEach { content ->
                        when ((content["type"] as? JsonPrimitive)?.content) {
                            "output_text" -> content["text"]?.let { add(buildJsonObject { put("text", it) }) }
                            "refusal" -> content["refusal"]?.let { add(buildJsonObject { put("text", it) }) }
                        }
                    }
                    "function_call" -> add(buildJsonObject {
                        putJsonObject("functionCall") {
                            put("name", item["name"] ?: JsonPrimitive(""))
                            put("id", item["call_id"] ?: item["id"] ?: JsonPrimitive("call_nutrime"))
                            val args = (item["arguments"] as? JsonPrimitive)?.content.orEmpty()
                            put("args", runCatching { json.parseToJsonElement(args) as JsonObject }.getOrElse { JsonObject(emptyMap()) })
                        }
                    })
                }
            }
        }
        if (parts.isEmpty()) throw LlmException("OpenAI returned no text or function call")
        return buildJsonObject {
            putJsonArray("candidates") {
                addJsonObject {
                    putJsonObject("content") { put("role", "model"); put("parts", parts) }
                    put("finishReason", "STOP")
                }
            }
        }
    }

    private fun normalizeSchema(value: JsonElement): JsonElement = when (value) {
        is JsonArray -> JsonArray(value.map(::normalizeSchema))
        is JsonObject -> buildJsonObject {
            value.forEach { (key, element) ->
                if (key == "format" && (element as? JsonPrimitive)?.content == "enum") return@forEach
                if (key == "type" && element is JsonPrimitive) put(key, element.content.lowercase())
                else put(key, normalizeSchema(element))
            }
        }
        else -> value
    }

    private fun errorMessage(body: String): String = runCatching {
        json.parseToJsonElement(body).jsonObject["error"]!!.jsonObject["message"]!!.jsonPrimitive.content
    }.getOrNull() ?: body.take(300)
}

/** Web nutrition lookup through OpenAI's Responses API web-search tool. */
class OpenAiWebNutritionClient(
    private val apiKey: String,
    private val model: String,
    private val baseUrl: String = "https://api.openai.com/v1",
) : WebNutritionClient {
    private val http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build()
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override suspend fun lookup(item: WebItem): WebAnswer {
        val body = buildJsonObject {
            put("model", model)
            put("input", prompt(item))
            putJsonArray("tools") { addJsonObject { put("type", "web_search") } }
            put("store", false)
            put("max_output_tokens", 1800)
            putJsonObject("reasoning") { put("effort", "low") }
        }
        val req = HttpRequest.newBuilder(URI("${baseUrl.trimEnd('/')}/responses"))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer $apiKey")
            .timeout(Duration.ofSeconds(60))
            .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
            .build()
        val res = withContext(Dispatchers.IO) { http.sendAsync(req, HttpResponse.BodyHandlers.ofString()).await() }
        if (res.statusCode() !in 200..299) {
            return WebAnswer(problem = if (res.statusCode() == 429) "The web lookup is busy." else "The web lookup failed.", error = true)
        }
        return parse(res.body())
    }

    internal fun parse(body: String): WebAnswer {
        val root = runCatching { json.parseToJsonElement(body).jsonObject }.getOrNull()
            ?: return WebAnswer(problem = "The web lookup gave no answer.", error = true)
        val text = (root["output"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }
            .flatMap { (it["content"] as? JsonArray).orEmpty() }
            .mapNotNull { (it as? JsonObject)?.get("text") as? JsonPrimitive }
            .joinToString("") { it.content }
        val start = text.indexOf('{')
        val end = text.lastIndexOf('}')
        if (start < 0 || end <= start) return WebAnswer(problem = "The web lookup gave no answer.", error = true)
        val obj = runCatching { json.parseToJsonElement(text.substring(start, end + 1)).jsonObject }.getOrNull()
            ?: return WebAnswer(problem = "The web lookup gave no answer.", error = true)
        if ((obj["found"] as? JsonPrimitive)?.content != "true") {
            return WebAnswer(problem = "Not found: " + ((obj["reason"] as? JsonPrimitive)?.content ?: "no published nutrition").take(200) + ".")
        }
        return runCatching { WebAnswer(found = json.decodeFromJsonElement(WebNutrition.serializer(), obj)) }
            .getOrElse { WebAnswer(problem = "Not found: incomplete numbers.") }
    }

    private fun prompt(item: WebItem): String = """
        Find published nutrition facts for this food using web search.
        Food: ${item.name}
        ${item.brand?.let { "Brand: $it" } ?: ""}
        ${item.restaurant?.let { "Restaurant: $it" } ?: ""}
        ${item.amount?.let { "Amount eaten (for choosing the size): $it" } ?: ""}

        Use the official brand or restaurant nutrition page or package label first, and prefer the Canadian page.
        Copy the source's numbers for one serving. Never estimate or calculate them. Include serving grams or ml
        only when published. If no reliable source contains the values, set found to false.

        Return ONLY one JSON object:
        {"found":true,"name":"...","brand":"... or null","serving_label":"1 sandwich","serving_grams":215,
        "calories":0,"protein_g":0,"carbs_g":0,"fat_g":0,"fiber_g":null,"sugars_g":null,
        "saturated_fat_g":null,"sodium_mg":null,"source_url":"https://...","source_note":"short source note"}
        or {"found":false,"reason":"short reason"}
    """.trimIndent()
}
