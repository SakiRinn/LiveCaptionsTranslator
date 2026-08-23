using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Security.Cryptography;
using System.Collections.Concurrent;
using System.Net.Http.Headers;


using LiveCaptionsTranslator.models;
using LiveCaptionsTranslator.utils;

namespace LiveCaptionsTranslator.apis
{
    public static class TranslateAPI
    {
        /*
         * The key of this field is used as the content for `translateAPIBox` in the `SettingPage`.
         * If you'd like to add a new API, please insert the key-value pair here.
         */
        public static readonly Dictionary<string, Func<string, CancellationToken, Task<string>>>
            TRANSLATE_FUNCTIONS = new()
        {
            { "Google", Google },
            { "Google2", Google2 },
            { "Ollama", Ollama },
            { "OpenAI", OpenAI },
            { "DeepL", DeepL },
            { "OpenRouter", OpenRouter },
            { "Youdao", Youdao },
            { "MTranServer", MTranServer },
            { "Baidu", Baidu },
            { "LibreTranslate", LibreTranslate },
        };
        public static readonly List<string> LLM_BASED_APIS = new()
        {
            "Ollama", "OpenAI", "OpenRouter"
        };
        public static readonly List<string> NO_CONFIG_APIS = new()
        {
            "Google", "Google2"
        };

        public static Func<string, CancellationToken, Task<string>> TranslateFunction =>
            TRANSLATE_FUNCTIONS[Translator.Setting.ApiName];
        public static bool IsLLMBased => LLM_BASED_APIS.Contains(Translator.Setting.ApiName);
        public static string Prompt => Translator.Setting.Prompt;

        private static readonly HttpClient client = new HttpClient()
        {
            Timeout = TimeSpan.FromSeconds(8)
        };

        private static readonly HttpClient openAIClient = new()
        {
            Timeout = System.Threading.Timeout.InfiniteTimeSpan
        };

        private static readonly ConcurrentDictionary<string, int>
            // This dictionary is used to store the fallback index for each OpenAI API URL.
            openAIFallbackIndexByUrl = new(StringComparer.OrdinalIgnoreCase);

        private const int OPENAI_TOTAL_TIMEOUT_SECONDS = 8;     // The total timeout for the entire OpenAI request, including retries and delays.
        private const int OPENAI_MAX_TRANSIENT_RETRIES = 2;     // This index is used to select the fallback model in case of transient errors.


        private static bool IsOfficialOpenAIEndpoint(Uri endpoint)
        {
            string host = endpoint.Host;

            return host.Equals(
                       "api.openai.com",
                       StringComparison.OrdinalIgnoreCase) ||
                   host.EndsWith(
                       ".api.openai.com",
                       StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsTransientOpenAIFailure(
            HttpStatusCode statusCode,
            string responseBody)
        {
            if (statusCode == HttpStatusCode.TooManyRequests)
            {
                // These 429 errors cannot be fixed by retrying the request.
                return !responseBody.Contains(
                           "credit_balance_exhausted",
                           StringComparison.OrdinalIgnoreCase) &&
                       !responseBody.Contains(
                           "spend_limit",
                           StringComparison.OrdinalIgnoreCase) &&
                       !responseBody.Contains(
                           "usage_limit",
                           StringComparison.OrdinalIgnoreCase);
            }

            int code = (int)statusCode;

            return statusCode == HttpStatusCode.RequestTimeout ||
                   code is 500 or 502 or 503 or 504;
        }

        private static TimeSpan GetOpenAIRetryDelay(
            HttpResponseMessage? response,
            int retryNumber)
        {
            TimeSpan delay = TimeSpan.FromMilliseconds(
                150 * Math.Pow(2, retryNumber));

            if (response?.Headers.RetryAfter?.Delta is TimeSpan retryAfter)
            {
                delay = retryAfter;
            }
            else if (response?.Headers.RetryAfter?.Date is DateTimeOffset retryDate)
            {
                delay = retryDate - DateTimeOffset.UtcNow;
            }

            if (delay < TimeSpan.Zero)
                delay = TimeSpan.Zero;

            // In live subtitles, it's not convenient to wait for too long.
            return delay > TimeSpan.FromSeconds(2)
                ? TimeSpan.FromSeconds(2)
                : delay;
        }

        private static string GetOpenAIError(
            HttpStatusCode statusCode,
            string responseBody)
        {
            string detail = responseBody.Trim();

            try
            {
                using var document = JsonDocument.Parse(responseBody);

                if (document.RootElement.TryGetProperty("error", out var error) &&
                    error.TryGetProperty("message", out var message))
                {
                    detail = message.GetString() ?? detail;
                }
            }
            catch (JsonException)
            {
                // Some compatible endpoints won't return JSON. We'll just use the raw response body as the error detail.
            }

            if (string.IsNullOrWhiteSpace(detail))
                detail = "The server returned no error details.";

            if (detail.Length > 500)
                detail = detail[..500] + "...";

            return $"[ERROR] Translation Failed: HTTP Error - " +
                   $"{statusCode}: {detail}";
        }




        public static async Task<string> OpenAI(string text, CancellationToken token = default){
            var config = Translator.Setting["OpenAI"] as OpenAIConfig;

            if (config == null)
                return "[ERROR] Translation Failed: OpenAI configuration not found.";

            if (string.IsNullOrWhiteSpace(config.ApiKey))
                return "[ERROR] Translation Failed: OpenAI API key is missing.";

            if (string.IsNullOrWhiteSpace(config.ModelName))
                return "[ERROR] Translation Failed: OpenAI model name is missing.";

            string apiUrl = TextUtil.NormalizeUrl(config.ApiUrl);

            if (!Uri.TryCreate(apiUrl, UriKind.Absolute, out var endpoint) ||
                (endpoint.Scheme != Uri.UriSchemeHttps &&
                 endpoint.Scheme != Uri.UriSchemeHttp))
            {
                return "[ERROR] Translation Failed: Invalid OpenAI API URL.";
            }

            bool isOfficialOpenAI = IsOfficialOpenAIEndpoint(endpoint);

            string language = OpenAIConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage,
                out var langValue)
                    ? langValue
                    : Translator.Setting.TargetLanguage;

            var messages = new List<BaseLLMConfig.Message>{
                            new BaseLLMConfig.Message
                            {
                                role = "system",
                                content = string.Format(Prompt, language)
                            },
                            new BaseLLMConfig.Message
                            {
                                role = "user",
                                content = $"🔤 {text} 🔤"
                            }
                        };

            if (Translator.Setting.ContextAware)
            {
                foreach (var entry in Translator.Caption.DisplayLogCards)
                {
                    string translatedText = entry.TranslatedText;

                    if (translatedText.Contains("[ERROR]") ||
                        translatedText.Contains("[WARNING]"))
                    {
                        continue;
                    }

                    translatedText = RegexPatterns.NoticePrefix()
                        .Replace(translatedText, "");

                    messages.InsertRange(
                        1,
                        [
                            new BaseLLMConfig.Message
                {
                    role = "user",
                    content = $"🔤 {entry.SourceText} 🔤"
                },
                new BaseLLMConfig.Message
                {
                    role = "assistant",
                    content = translatedText
                }
                        ]);
                }
            }

            int fallbackCount = LLMRequestDataFactory.FallbackCount;

            int fallbackIndex = isOfficialOpenAI
                ? 0
                : openAIFallbackIndexByUrl.GetOrAdd(apiUrl, 0);

            if (fallbackIndex < 0 || fallbackIndex >= fallbackCount)
                fallbackIndex = 0;

            int maximumSchemas = isOfficialOpenAI ? 1 : fallbackCount;
            int schemasTried = 0;
            int transientRetries = 0;

            string lastError =
                "[ERROR] Translation Failed: No compatible request format found.";

            using var timeoutSource =
                CancellationTokenSource.CreateLinkedTokenSource(token);

            timeoutSource.CancelAfter(
                TimeSpan.FromSeconds(OPENAI_TOTAL_TIMEOUT_SECONDS));

            CancellationToken requestToken = timeoutSource.Token;

            try
            {
                while (schemasTried < maximumSchemas)
                {
                    requestToken.ThrowIfCancellationRequested();

                    object requestData;

                    if (isOfficialOpenAI)
                    {
                        // Specific body for POST /v1/chat/completions.
                        requestData = new
                        {
                            model = config.ModelName,
                            messages,
                            temperature = config.Temperature,
                            max_completion_tokens = 256,
                            reasoning_effort = "none",
                            stream = false
                        };
                    }
                    else
                    {
                        // Preserves the original mechanism for compatible APIs.
                        requestData = LLMRequestDataFactory.Create(
                            fallbackIndex,
                            config.ModelName,
                            messages,
                            config.Temperature);
                    }

                    string jsonContent = JsonSerializer.Serialize(
                        requestData,
                        requestData.GetType());

                    using var request = new HttpRequestMessage(
                        HttpMethod.Post,
                        endpoint);

                    request.Headers.Authorization =
                        new AuthenticationHeaderValue(
                            "Bearer",
                            config.ApiKey);

                    request.Content = new StringContent(
                        jsonContent,
                        Encoding.UTF8,
                        "application/json");

                    HttpResponseMessage response;

                    try
                    {
                        response = await openAIClient.SendAsync(
                            request,
                            HttpCompletionOption.ResponseContentRead,
                            requestToken);
                    }
                    catch (HttpRequestException)
                        when (transientRetries < OPENAI_MAX_TRANSIENT_RETRIES)
                    {
                        TimeSpan delay = GetOpenAIRetryDelay(
                            null,
                            transientRetries);

                        transientRetries++;

                        await Task.Delay(delay, requestToken);
                        continue;
                    }

                    using (response)
                    {
                        string responseBody =
                            await response.Content.ReadAsStringAsync(requestToken);

                        if (response.IsSuccessStatusCode)
                        {
                            if (!isOfficialOpenAI)
                            {
                                openAIFallbackIndexByUrl[apiUrl] =
                                    fallbackIndex;
                            }

                            var responseObject =
                                JsonSerializer.Deserialize<OpenAIConfig.Response>(
                                    responseBody);

                            string? output = responseObject?
                                .choices?
                                .FirstOrDefault()?
                                .message?
                                .content;

                            if (string.IsNullOrWhiteSpace(output))
                            {
                                return "[ERROR] Translation Failed: " +
                                       "Unexpected OpenAI response format.";
                            }

                            return RegexPatterns.ModelThinking()
                                .Replace(output, "");
                        }

                        lastError = GetOpenAIError(
                            response.StatusCode,
                            responseBody);

                        if (IsTransientOpenAIFailure(
                                response.StatusCode,
                                responseBody) &&
                            transientRetries < OPENAI_MAX_TRANSIENT_RETRIES)
                        {
                            TimeSpan delay = GetOpenAIRetryDelay(
                                response,
                                transientRetries);

                            transientRetries++;

                            await Task.Delay(delay, requestToken);
                            continue;
                        }

                        bool shouldTryAnotherSchema =
                            !isOfficialOpenAI &&
                            (response.StatusCode == HttpStatusCode.BadRequest ||
                             response.StatusCode ==
                                 HttpStatusCode.UnprocessableEntity);

                        if (!shouldTryAnotherSchema)
                            return lastError;

                        schemasTried++;

                        if (schemasTried >= maximumSchemas)
                            break;

                        fallbackIndex =
                            (fallbackIndex + 1) % fallbackCount;

                        transientRetries = 0;
                    }
                }

                return lastError;
            }
            catch (OperationCanceledException)
                when (!token.IsCancellationRequested)
            {
                return "[ERROR] Translation Failed: " +
                       "The complete request exceeded 8 seconds.";
            }
            catch (OperationCanceledException)
            {
                // Obsolete translation canceled by TranslationTaskQueue.
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }
        }

        public static async Task<string> Ollama(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["Ollama"] as OllamaConfig;
            string language = OllamaConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;
            string apiUrl = TextUtil.NormalizeUrl(config.ApiUrl + "/api/chat");

            var messages = new List<BaseLLMConfig.Message>
            {
                new BaseLLMConfig.Message { role = "system", content = string.Format(Prompt, language) },
                new BaseLLMConfig.Message { role = "user", content = $"🔤 {text} 🔤" }
            };
            if (Translator.Setting.ContextAware)
            {
                foreach (var entry in Translator.Caption.DisplayLogCards)
                {
                    string translatedText = entry.TranslatedText;
                    if (translatedText.Contains("[ERROR]") || translatedText.Contains("[WARNING]"))
                        continue;
                    translatedText = RegexPatterns.NoticePrefix().Replace(translatedText, "");

                    messages.InsertRange(1, [
                        new BaseLLMConfig.Message { role = "user", content = $"🔤 {entry.SourceText} 🔤" },
                        new BaseLLMConfig.Message { role = "assistant", content = $"{translatedText}" }
                    ]);
                }
            }

            var requestData = LLMRequestDataFactory.Create("Ollama", config.ModelName, messages, config.Temperature);

            string jsonContent = JsonSerializer.Serialize(requestData, requestData.GetType());
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            client.DefaultRequestHeaders.Clear();

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(apiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();
                var responseObj = JsonSerializer.Deserialize<OllamaConfig.Response>(responseString);
                var output = responseObj.message.content;
                return RegexPatterns.ModelThinking().Replace(output, "");
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }

        public static async Task<string> OpenRouter(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["OpenRouter"] as OpenRouterConfig;
            string language = OpenRouterConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;
            string apiUrl = "https://openrouter.ai/api/v1/chat/completions";

            var messages = new List<BaseLLMConfig.Message>
            {
                new BaseLLMConfig.Message { role = "system", content = string.Format(Prompt, language) },
                new BaseLLMConfig.Message { role = "user", content = $"🔤 {text} 🔤" }
            };
            if (Translator.Setting.ContextAware)
            {
                foreach (var entry in Translator.Caption.DisplayLogCards)
                {
                    string translatedText = entry.TranslatedText;
                    if (translatedText.Contains("[ERROR]") || translatedText.Contains("[WARNING]"))
                        continue;
                    translatedText = RegexPatterns.NoticePrefix().Replace(translatedText, "");

                    messages.InsertRange(1, [
                        new BaseLLMConfig.Message { role = "user", content = $"🔤 {entry.SourceText} 🔤" },
                        new BaseLLMConfig.Message { role = "assistant", content = $"{translatedText}" }
                    ]);
                }
            }

            var requestData = LLMRequestDataFactory.Create("OpenRouter", config.ModelName, messages, config.Temperature);

            string jsonContent = JsonSerializer.Serialize(requestData, requestData.GetType());
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            client.DefaultRequestHeaders.Clear();
            client.DefaultRequestHeaders.Add("Authorization", $"Bearer {config?.ApiKey}");

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(apiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync();
                var jsonResponse = JsonSerializer.Deserialize<JsonElement>(responseContent);
                var output = jsonResponse.GetProperty("choices")[0]
                                         .GetProperty("message")
                                         .GetProperty("content")
                                         .GetString() ?? string.Empty;
                return RegexPatterns.ModelThinking().Replace(output, "");
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }

        public static async Task<string> Google(string text, CancellationToken token = default)
        {
            var language = Translator.Setting?.TargetLanguage;

            string encodedText = Uri.EscapeDataString(text);
            var url = $"https://clients5.google.com/translate_a/t?" +
                      $"client=dict-chrome-ex&sl=auto&" +
                      $"tl={language}&" +
                      $"q={encodedText}";

            HttpResponseMessage response;
            try
            {
                response = await client.GetAsync(url, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();

                var responseObj = JsonSerializer.Deserialize<List<List<string>>>(responseString);

                string translatedText = responseObj[0][0];
                return translatedText;
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }

        public static async Task<string> Google2(string text, CancellationToken token = default)
        {
            string apiKey = "AIzaSyA6EEtrDCfBkHV8uU2lgGY-N383ZgAOo7Y";
            var language = Translator.Setting?.TargetLanguage;
            string strategy = "2";

            string encodedText = Uri.EscapeDataString(text);
            string url = $"https://dictionaryextension-pa.googleapis.com/v1/dictionaryExtensionData?" +
                         $"language={language}&" +
                         $"key={apiKey}&" +
                         $"term={encodedText}&" +
                         $"strategy={strategy}";

            var request = new HttpRequestMessage(HttpMethod.Get, url);
            request.Headers.Add("x-referer", "chrome-extension://mgijmajocgfcbeboacabfgobmjgjcoja");

            HttpResponseMessage response;
            try
            {
                response = await client.SendAsync(request, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseBody = await response.Content.ReadAsStringAsync();

                using var jsonDoc = JsonDocument.Parse(responseBody);
                var root = jsonDoc.RootElement;

                if (root.TryGetProperty("translateResponse", out JsonElement translateResponse))
                {
                    string translatedText = translateResponse.GetProperty("translateText").GetString();
                    return translatedText;
                }
                else
                    return "[ERROR] Translation Failed: Unexpected API response format";
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }

        public static async Task<string> DeepL(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["DeepL"] as DeepLConfig;
            string language = DeepLConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;
            string apiUrl = TextUtil.NormalizeUrl(config.ApiUrl);

            var requestData = new
            {
                text = new[] { text },
                target_lang = language
            };

            string jsonContent = JsonSerializer.Serialize(requestData);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            client.DefaultRequestHeaders.Clear();
            client.DefaultRequestHeaders.Add("Authorization", $"DeepL-Auth-Key {config?.ApiKey}");

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(apiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(responseString);

                if (doc.RootElement.TryGetProperty("translations", out var translations) &&
                    translations.ValueKind == JsonValueKind.Array && translations.GetArrayLength() > 0)
                {
                    return translations[0].GetProperty("text").GetString();
                }
                return "[ERROR] Translation Failed: No valid feedback";
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }


        public static async Task<string> Youdao(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["Youdao"] as YoudaoConfig;
            string language = YoudaoConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;

            string salt = DateTime.Now.Millisecond.ToString();
            string sign = BitConverter.ToString(
                MD5.Create().ComputeHash(
                    Encoding.UTF8.GetBytes($"{config.AppKey}{text}{salt}{config.AppSecret}"))).Replace("-", "").ToLower();

            var parameters = new Dictionary<string, string>
            {
                ["q"] = text,
                ["from"] = "auto",
                ["to"] = language,
                ["appKey"] = config.AppKey,
                ["salt"] = salt,
                ["sign"] = sign
            };

            var content = new FormUrlEncodedContent(parameters);
            client.DefaultRequestHeaders.Clear();

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(config.ApiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();
                var responseObj = JsonSerializer.Deserialize<YoudaoConfig.TranslationResult>(responseString);

                if (responseObj.errorCode != "0")
                    return $"[ERROR] Translation Failed: Youdao Error - {responseObj.errorCode}";

                return responseObj.translation?.FirstOrDefault() ?? "[ERROR] Translation Failed: No content";
            }
            else
            {
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
            }
        }

        public static async Task<string> MTranServer(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["MTranServer"] as MTranServerConfig;
            string targetLanguage = MTranServerConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;
            string sourceLanguage = config.SourceLanguage;
            string apiUrl = TextUtil.NormalizeUrl(config.ApiUrl);

            var requestData = new
            {
                text = text,
                to = targetLanguage,
                from = sourceLanguage
            };

            string jsonContent = JsonSerializer.Serialize(requestData);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            client.DefaultRequestHeaders.Clear();
            client.DefaultRequestHeaders.Add("Authorization", $"Bearer {config?.ApiKey}");

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(apiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();
                var responseObj = JsonSerializer.Deserialize<MTranServerConfig.Response>(responseString);
                return responseObj.result;
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }

        public static async Task<string> Baidu(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["Baidu"] as BaiduConfig;
            string language = BaiduConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;

            string salt = DateTime.Now.Millisecond.ToString();
            string sign = BitConverter.ToString(
                MD5.Create().ComputeHash(
                    Encoding.UTF8.GetBytes($"{config.AppId}{text}{salt}{config.AppSecret}"))).Replace("-", "").ToLower();

            var parameters = new Dictionary<string, string>
            {
                ["q"] = text,
                ["from"] = "auto",
                ["to"] = language,
                ["appid"] = config.AppId,
                ["salt"] = salt,
                ["sign"] = sign
            };

            var content = new FormUrlEncodedContent(parameters);
            client.DefaultRequestHeaders.Clear();

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(config.ApiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();
                var responseObj = JsonSerializer.Deserialize<BaiduConfig.TranslationResult>(responseString);

                if (responseObj.error_code is not null && responseObj.error_code != "0")
                    return $"[ERROR] Translation Failed: Baidu Error - {responseObj.error_code}";

                return responseObj.trans_result?.FirstOrDefault()?.dst ?? "[ERROR] Translation Failed: No content";
            }
            else
            {
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
            }
        }

        public static async Task<string> LibreTranslate(string text, CancellationToken token = default)
        {
            var config = Translator.Setting["LibreTranslate"] as LibreTranslateConfig;
            string targetLanguage = LibreTranslateConfig.SupportedLanguages.TryGetValue(
                Translator.Setting.TargetLanguage, out var langValue) ? langValue : Translator.Setting.TargetLanguage;
            string apiUrl = TextUtil.NormalizeUrl(config.ApiUrl);

            var requestData = new
            {
                q = text,
                target = targetLanguage,
                source = "auto",
                format = "text",
                api_key = config?.ApiKey
            };

            string jsonContent = JsonSerializer.Serialize(requestData);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            client.DefaultRequestHeaders.Clear();

            HttpResponseMessage response;
            try
            {
                response = await client.PostAsync(apiUrl, content, token);
            }
            catch (OperationCanceledException ex)
            {
                if (ex.Message.StartsWith("The request"))
                    return $"[ERROR] Translation Failed: The request was canceled due to timeout (> 8 seconds), " +
                           $"please use a faster API or check network connection.";
                throw;
            }
            catch (Exception ex)
            {
                return $"[ERROR] Translation Failed: {ex.Message}";
            }

            if (response.IsSuccessStatusCode)
            {
                string responseString = await response.Content.ReadAsStringAsync();
                var responseObj = JsonSerializer.Deserialize<LibreTranslateConfig.Response>(responseString);
                return responseObj.translatedText;
            }
            else
                return $"[ERROR] Translation Failed: HTTP Error - {response.StatusCode}";
        }
    }

    public class ConfigDictConverter : JsonConverter<Dictionary<string, List<TranslateAPIConfig>>>
    {
        public override Dictionary<string, List<TranslateAPIConfig>> Read(
            ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected a StartObject token.");
            var configs = new Dictionary<string, List<TranslateAPIConfig>>();

            reader.Read();
            while (reader.TokenType == JsonTokenType.PropertyName)
            {
                string key = reader.GetString();
                reader.Read();

                var configType = Type.GetType($"LiveCaptionsTranslator.models.{key}Config");
                TranslateAPIConfig config;

                if (reader.TokenType == JsonTokenType.StartArray)
                {
                    var list = new List<TranslateAPIConfig>();
                    reader.Read();

                    while (reader.TokenType != JsonTokenType.EndArray)
                    {
                        if (configType != null && typeof(TranslateAPIConfig).IsAssignableFrom(configType))
                            config = (TranslateAPIConfig)JsonSerializer.Deserialize(ref reader, configType, options);
                        else
                            config = (TranslateAPIConfig)JsonSerializer.Deserialize(ref reader, typeof(TranslateAPIConfig), options);

                        list.Add(config);
                        reader.Read();
                    }
                    configs[key] = list;
                }
                else
                    throw new JsonException("Expected a StartObject token or a StartArray token.");

                reader.Read();
            }

            if (reader.TokenType != JsonTokenType.EndObject)
                throw new JsonException("Expected an EndObject token.");
            return configs;
        }

        public override void Write(
            Utf8JsonWriter writer, Dictionary<string, List<TranslateAPIConfig>> value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            foreach (var kvp in value)
            {
                writer.WritePropertyName(kvp.Key);
                var configType = Type.GetType($"LiveCaptionsTranslator.models.{kvp.Key}Config");

                if (kvp.Value is IEnumerable<TranslateAPIConfig> configList)
                {
                    writer.WriteStartArray();
                    foreach (var config in configList)
                    {
                        if (configType != null && typeof(TranslateAPIConfig).IsAssignableFrom(configType))
                            JsonSerializer.Serialize(writer, config, configType, options);
                        else
                            JsonSerializer.Serialize(writer, config, typeof(TranslateAPIConfig), options);
                    }
                    writer.WriteEndArray();
                }
                else
                    throw new JsonException($"Unsupported config type: {kvp.Value.GetType()}");
            }
            writer.WriteEndObject();
        }
    }
}
