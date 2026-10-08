/*
Purpose: Option registry of APD. One table describes every run-time option (name, type, default,
         range, unit, help, pipeline stage). The same table feeds the command line, the INI config
         file, the "effective configuration" dump and the JSON schema a GUI can build widgets from.
Status:  Testing
Future:  Per-option GUI hints (widget type, advanced flag) once the Recova GUI consumes the schema.
*/
#ifndef _APD_CONFIG_H_
#define _APD_CONFIG_H_

#include "main.h"

// Storage type of one option; decides parsing, printing and the JSON "type" field.
enum OptionType {
    OPTION_INT,
    OPTION_INT64,
    OPTION_FLOAT,
    OPTION_DOUBLE,
    OPTION_BOOL,
    OPTION_ENUM
};

// One row of the registry.
struct OptionSpec {
    std::string name;                  // "group.key", e.g. "fusion.depth_mode"
    OptionType type;                   // storage type of *target
    void *target;                      // address of the field inside the RuntimeConfig
    bool has_range;                    // true when min_value/max_value are enforced
    double min_value;                  // inclusive lower bound
    double max_value;                  // inclusive upper bound
    std::vector<std::string> choices;  // names of the enum values, index = stored int
    std::string stage;                 // "patchmatch": needs a full run; "fusion": --only_fuse is enough
    std::string unit;                  // free text, e.g. "px", "rad", "deg", "world"
    std::string help;                  // one-line description
    std::string default_text;          // value captured at registration, before any override
};

class OptionRegistry {
public:
    // Registers every option and records the defaults currently stored in cfg.
    explicit OptionRegistry(RuntimeConfig &cfg);

    // Adds "--group.key <value>" for every option to a boost options description.
    void AddToDescription(boost::program_options::options_description &desc) const;

    // Writes every option present in vm into the RuntimeConfig; returns false and fills error on bad input.
    bool Apply(const boost::program_options::variables_map &vm, std::string &error);

    // Cross-option checks that a single range cannot express.
    bool Validate(std::string &error) const;

    // Machine-readable description of all options (for GUIs).
    std::string ToJson() const;

    // Current values as an INI file that --config can read back.
    std::string ToIni() const;

    // Names of the options of one group ("fusion", ...) whose current value differs from the default.
    std::vector<std::string> ChangedOptions(const std::string &group) const;

private:
    void Add(const std::string &name, OptionType type, void *target, bool has_range, double min_value,
             double max_value, const std::string &stage, const std::string &unit, const std::string &help,
             const std::vector<std::string> &choices = std::vector<std::string>());

    std::string ValueText(const OptionSpec &spec) const;

    bool SetFromText(const OptionSpec &spec, const std::string &text, std::string &error) const;

    RuntimeConfig &config;
    std::vector<OptionSpec> specs;
};

#endif // !_APD_CONFIG_H_
