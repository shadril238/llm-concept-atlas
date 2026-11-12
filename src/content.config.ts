import { defineCollection } from "astro:content";
import { z } from "astro/zod";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        kind: z.enum(["part", "chapter", "concept", "atlas", "path", "use-case"]).optional(),
        conceptId: z
          .string()
          .regex(/^[a-z0-9-]+$/)
          .optional(),
        chapter: z.number().int().min(1).max(12).optional(),
        part: z.enum(["foundations", "applications", "training"]).optional(),
        summary: z.string().optional(),
        parent: z.string().optional(),
        prerequisites: z.array(z.string()).default([]),
        related: z.array(z.string()).default([]),
        level: z.enum(["beginner", "intermediate", "advanced"]).optional(),
        lessonStatus: z.enum(["primer", "full"]).default("primer"),
        tags: z.array(z.string()).default([]),
        lastReviewed: z.coerce.date().optional(),
      }),
    }),
  }),
};
