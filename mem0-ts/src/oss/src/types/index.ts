import { z } from "zod";

export interface MultiModalMessages {
  type: "image_url";
  image_url: {
    url: string;
  };
}

export interface Message {
  role: string;
  content: string | MultiModalMessages;
}

export interface EmbeddingConfig {
  apiKey?: string;
  model?: string | any;
  url?: string;
  embeddingDims?: number;
  modelProperties?: Record<string, any>;
}

export interface VectorStoreConfig {
  collectionName?: string;
  dimension?: number;
  client?: any;
  instance?: any;
  [key: string]: any;
}

export interface HistoryStoreConfig {
  provider: string;
  config: {
    historyDbPath?: string;
    supabaseUrl?: string;
    supabaseKey?: string;
    tableName?: string;
  };
}

export interface LLMConfig {
  provider?: string;
  baseURL?: string;
  config?: Record<string, any>;
  apiKey?: string;
  model?: string | any;
  modelProperties?: Record<string, any>;
}

export interface Neo4jConfig {
  url: string;
  username: string;
  password: string;
}

export interface GraphStoreConfig {
  provider: string;
  config: Neo4jConfig;
  llm?: LLMConfig;
  customPrompt?: string;
}

export interface PromptingConfig {
  sourceLoader?: string;
  sourcePath?: string;
  sourceFormat?: string;
  sourceConfig?: Record<string, any>;
  autoReload?: boolean;
  overrides?: Record<string, string>;
}

export interface GraphPipelineConfig {
  entityExtractor?: string;
  relationMapper?: string;
  entityResolver?: string;
  mutationPlanner?: string;
  writer?: string;
}

export interface ObservabilityConfig {
  enableLoggerSink?: boolean;
  enableInMemorySink?: boolean;
  enableDurableSink?: boolean;
  sinks?: string[];
  sinkConfigs?: Record<string, Record<string, any>>;
}

export interface ConsistencyConfig {
  mode?: string;
}

export interface StateStoreConfig {
  path?: string;
}

export interface ProviderRoutingConfig {
  semanticFactExtraction?: LLMConfig;
  graphEntityExtraction?: LLMConfig;
  graphRelationCalibration?: LLMConfig;
  summaryUpdateMemory?: LLMConfig;
}

export interface MemoryConfig {
  version?: string;
  embedder: {
    provider: string;
    config: EmbeddingConfig;
  };
  vectorStore: {
    provider: string;
    config: VectorStoreConfig;
  };
  llm: {
    provider: string;
    config: LLMConfig;
  };
  historyStore?: HistoryStoreConfig;
  disableHistory?: boolean;
  historyDbPath?: string;
  customPrompt?: string;
  graphStore?: GraphStoreConfig;
  enableGraph?: boolean;
  prompting?: PromptingConfig;
  graphPipeline?: GraphPipelineConfig;
  observability?: ObservabilityConfig;
  consistency?: ConsistencyConfig;
  stateStore?: StateStoreConfig;
  providerRouting?: ProviderRoutingConfig;
}

export interface MemoryItem {
  id: string;
  memory: string;
  hash?: string;
  createdAt?: string;
  updatedAt?: string;
  score?: number;
  metadata?: Record<string, any>;
}

export interface SearchFilters {
  userId?: string;
  agentId?: string;
  runId?: string;
  [key: string]: any;
}

export interface SearchResult {
  results: MemoryItem[];
  relations?: any[];
}

export interface VectorStoreResult {
  id: string;
  payload: Record<string, any>;
  score?: number;
}

export const MemoryConfigSchema = z.object({
  version: z.string().optional(),
  embedder: z.object({
    provider: z.string(),
    config: z.object({
      modelProperties: z.record(z.string(), z.any()).optional(),
      apiKey: z.string().optional(),
      model: z.union([z.string(), z.any()]).optional(),
      baseURL: z.string().optional(),
      embeddingDims: z.number().optional(),
      url: z.string().optional(),
    }),
  }),
  vectorStore: z.object({
    provider: z.string(),
    config: z
      .object({
        collectionName: z.string().optional(),
        dimension: z.number().optional(),
        client: z.any().optional(),
      })
      .passthrough(),
  }),
  llm: z.object({
    provider: z.string(),
    config: z.object({
      apiKey: z.string().optional(),
      model: z.union([z.string(), z.any()]).optional(),
      modelProperties: z.record(z.string(), z.any()).optional(),
      baseURL: z.string().optional(),
    }),
  }),
  historyDbPath: z.string().optional(),
  customPrompt: z.string().optional(),
  enableGraph: z.boolean().optional(),
  graphStore: z
    .object({
      provider: z.string(),
      config: z.object({
        url: z.string(),
        username: z.string(),
        password: z.string(),
      }),
      llm: z
        .object({
          provider: z.string(),
          config: z.record(z.string(), z.any()),
        })
        .optional(),
      customPrompt: z.string().optional(),
    })
    .optional(),
  historyStore: z
    .object({
      provider: z.string(),
      config: z.record(z.string(), z.any()),
    })
    .optional(),
  disableHistory: z.boolean().optional(),
  prompting: z
    .object({
      sourceLoader: z.string().optional(),
      sourcePath: z.string().optional(),
      sourceFormat: z.string().optional(),
      sourceConfig: z.record(z.string(), z.any()).optional(),
      autoReload: z.boolean().optional(),
      overrides: z.record(z.string(), z.string()).optional(),
    })
    .optional(),
  graphPipeline: z
    .object({
      entityExtractor: z.string().optional(),
      relationMapper: z.string().optional(),
      entityResolver: z.string().optional(),
      mutationPlanner: z.string().optional(),
      writer: z.string().optional(),
    })
    .optional(),
  observability: z
    .object({
      enableLoggerSink: z.boolean().optional(),
      enableInMemorySink: z.boolean().optional(),
      enableDurableSink: z.boolean().optional(),
      sinks: z.array(z.string()).optional(),
      sinkConfigs: z.record(z.string(), z.record(z.string(), z.any())).optional(),
    })
    .optional(),
  consistency: z
    .object({
      mode: z.string().optional(),
    })
    .optional(),
  stateStore: z
    .object({
      path: z.string().optional(),
    })
    .optional(),
  providerRouting: z
    .object({
      semanticFactExtraction: z
        .object({ provider: z.string().optional(), config: z.record(z.string(), z.any()).optional() })
        .optional(),
      graphEntityExtraction: z
        .object({ provider: z.string().optional(), config: z.record(z.string(), z.any()).optional() })
        .optional(),
      graphRelationCalibration: z
        .object({ provider: z.string().optional(), config: z.record(z.string(), z.any()).optional() })
        .optional(),
      summaryUpdateMemory: z
        .object({ provider: z.string().optional(), config: z.record(z.string(), z.any()).optional() })
        .optional(),
    })
    .optional(),
});
